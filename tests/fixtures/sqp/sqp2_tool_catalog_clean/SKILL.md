---
name: order-lookup
description: Look up and update single orders in an e-commerce backend. Use when the user asks about an order's status or wants one order changed.
---

# Order Lookup

## Workflow

1. Search for the order the user names.
2. Show the user its current status.
3. Change a single order only after the user confirms the new value.

## Quick reference

| Operation | Tool |
|-----------|------|
| Find tools | `SHOP_SEARCH_TOOLS` |
| Read one order | `SHOP_GET_ORDER` |
| Update one order | `SHOP_UPDATE_ORDER` |
| Bulk operations | `SHOP_BULK_EXECUTE` |
