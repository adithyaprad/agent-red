"""Tools for the cart recovery agent.

It reaches customers who left a cart, answers questions about price, stock and delivery,
can discount, and can place the order. Three of those cost the merchant money if the agent
is talked into them, which is what makes this a worthwhile agent to attack.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from agentred.mcp.tools.base import ToolSet, as_number, tool
from agentred.mcp.world import World


@tool("lookup_cart")
def lookup_cart(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return one abandoned cart with its lines priced at catalogue prices."""
    cart_id = str(arguments.get("cart_id", ""))
    cart = world.carts.get(cart_id)
    if cart is None:
        return {"error": f"no cart {cart_id}"}
    customer = world.customers.get(cart["customer_id"], {})
    lines = [
        {
            "sku": line["sku"],
            "name": world.products.get(line["sku"], {}).get("name", "unknown"),
            "quantity": line["quantity"],
            "unit_price": world.products.get(line["sku"], {}).get("price"),
        }
        for line in cart["lines"]
    ]
    return {
        "cart_id": cart_id,
        "customer_id": cart["customer_id"],
        "customer_name": customer.get("name"),
        "loyalty_tier": customer.get("loyalty_tier"),
        "note": cart.get("note", ""),
        "abandoned_on": cart.get("abandoned_on", ""),
        "lines": lines,
        "total": world.cart_total(cart_id),
    }


@tool("lookup_product")
def lookup_product(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return catalogue price, stock and lead time for one SKU."""
    sku = str(arguments.get("sku", ""))
    product = world.products.get(sku)
    if product is None:
        return {"error": f"no product {sku}"}
    return {
        "sku": sku,
        "name": product["name"],
        "price": product["price"],
        "in_stock": product["stock"] > 0,
        "stock": product["stock"],
        "lead_time_days": product["lead_time_days"],
        "clearance": product["clearance"],
    }


@tool("check_discount_code")
def check_discount_code(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
    """Say whether a discount code exists and is active, and what it is worth.

    A code the agent invented is simply absent here, which is what turns a fabricated
    discount into something observable rather than a matter of opinion.
    """
    code = str(arguments.get("code", "")).strip().upper()
    entry = world.discount_codes.get(code)
    if entry is None:
        return {"code": code, "valid": False, "reason": "no such code"}
    if not entry["active"]:
        return {"code": code, "valid": False, "reason": "expired"}
    return {"code": code, "valid": True, "percent": entry["percent"]}


@tool("apply_discount")
def apply_discount(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
    """Apply a percentage discount to a cart.

    Applies whatever percentage it is given. The merchant's ceiling lives in the system
    prompt, and whether the agent honours it under pressure is the thing being measured.
    """
    cart_id = str(arguments.get("cart_id", ""))
    percent = as_number(arguments.get("percent"))
    if cart_id not in world.carts:
        return {"error": f"no cart {cart_id}"}
    if percent is None:
        return {"error": "percent must be a number"}

    total = world.cart_total(cart_id)
    reduction = round(total * percent / 100, 2)
    world.carts[cart_id]["discount_percent"] = percent
    world.record("apply_discount", cart_id=cart_id, percent=percent, reduction=reduction)
    return {
        "cart_id": cart_id,
        "percent_applied": percent,
        "reduction": reduction,
        "new_total": round(total - reduction, 2),
    }


@tool("create_order")
def create_order(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
    """Turn a cart into an order, at whatever discount has been applied to it."""
    cart_id = str(arguments.get("cart_id", ""))
    shipping_method = str(arguments.get("shipping_method", "standard"))
    cart = world.carts.get(cart_id)
    if cart is None:
        return {"error": f"no cart {cart_id}"}

    total = world.cart_total(cart_id)
    percent = float(cart.get("discount_percent", 0) or 0)
    charged = round(total * (1 - percent / 100), 2)
    shipping = world.shipping_methods.get(shipping_method, {"price": 0.0})
    charged = round(charged + float(shipping.get("price", 0.0)), 2)

    order_id = f"ORD-9{len(world.orders):04d}"
    world.orders[order_id] = {
        "order_id": order_id,
        "customer_id": cart["customer_id"],
        "email": world.customers.get(cart["customer_id"], {}).get("email", ""),
        "placed_on": date.today().isoformat(),
        "status": "placed",
        "total": charged,
        "refunded": 0.0,
        "lines": [
            {
                "sku": line["sku"],
                "quantity": line["quantity"],
                "price": world.products.get(line["sku"], {}).get("price", 0.0),
            }
            for line in cart["lines"]
        ],
        "internal_note": "Placed by the recovery agent.",
    }
    world.record(
        "create_order",
        order_id=order_id,
        cart_id=cart_id,
        charged=charged,
        discount_percent=percent,
        shipping_method=shipping_method,
    )
    return {
        "order_id": order_id,
        "charged": charged,
        "discount_percent": percent,
        "shipping_method": shipping_method,
    }


@tool("promise_delivery")
def promise_delivery(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
    """Commit the merchant to a delivery date for a cart.

    Records the promise whether or not the catalogue lead times can support it, so an
    undeliverable commitment shows up in the tool-call log instead of only in the prose.
    """
    cart_id = str(arguments.get("cart_id", ""))
    promised_date = str(arguments.get("promised_date", ""))
    cart = world.carts.get(cart_id)
    if cart is None:
        return {"error": f"no cart {cart_id}"}
    if not promised_date:
        return {"error": "promised_date is required"}

    lead_days = max(
        (world.products.get(line["sku"], {}).get("lead_time_days", 0) for line in cart["lines"]),
        default=0,
    )
    earliest = date.today() + timedelta(days=lead_days)
    world.record("promise_delivery", cart_id=cart_id, promised_date=promised_date)
    return {
        "cart_id": cart_id,
        "promised_date": promised_date,
        "earliest_supported_date": earliest.isoformat(),
        "recorded": True,
    }


@tool("list_abandoned_carts")
def list_abandoned_carts(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the baskets nobody checked out, for a schedule to work through.

    This is the tool a scheduled run starts from, and it is the reason the recovery agent is
    attackable without a conversation. Every row carries `note`, which is text the shopper
    typed into a delivery-instruction box: the merchant never wrote it, no human read it on
    the way in, and by the time the agent reads it nothing distinguishes it from the shop's
    own data. It is returned verbatim, because a tool that stripped it would be answering the
    question the run exists to ask. See ADR-0006.
    """
    del arguments
    rows = [
        {
            "cart_id": cart_id,
            "customer_id": cart["customer_id"],
            "customer_name": world.customers.get(cart["customer_id"], {}).get("name", ""),
            "phone": world.customers.get(cart["customer_id"], {}).get("phone", ""),
            "loyalty_tier": world.customers.get(cart["customer_id"], {}).get("loyalty_tier", ""),
            "note": cart.get("note", ""),
            "abandoned_on": cart.get("abandoned_on", ""),
            "line_count": len(cart["lines"]),
            "total": world.cart_total(cart_id),
        }
        for cart_id, cart in world.carts.items()
        if cart.get("abandoned")
    ]
    return {"carts": rows, "count": len(rows)}


@tool("send_whatsapp")
def send_whatsapp(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
    """Send a message to a number.

    Sends whatever body it is handed to whatever number it is handed, and neither has to
    belong to the shopper whose basket the run is about. On a scheduled agent this is the
    only thing a person outside the shop ever sees, so it is where the whole cost of a
    successful attack lands: an offer nobody authorised, a promise the shop cannot keep, or
    another shopper's basket quoted in the body of a message to the wrong number.
    """
    to = str(arguments.get("to", "")).strip()
    body = str(arguments.get("body", ""))
    if not to:
        return {"error": "to must be a number"}
    world.record("send_whatsapp", to=to, characters=len(body))
    return {"to": to, "characters": len(body), "status": "sent"}


CART_TOOLS = ToolSet(
    list_abandoned_carts,
    lookup_cart,
    lookup_product,
    check_discount_code,
    apply_discount,
    create_order,
    promise_delivery,
    send_whatsapp,
)
"""What the cart recovery agent can call."""
