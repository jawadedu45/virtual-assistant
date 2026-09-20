# What changed, and what you need to do

## 1. Pictures now get sent to the customer (`main.py`, `app.js`)
- New `find_product_media()` in `main.py` returns **both** the image and the video for a matched product (it replaces the old `find_product_video()`).
- `/chat` and `/voice-chat` now return `image_url` alongside `video_url` when a product's keyword matches the customer's message.
- `app.js` has a new `addImageBubble()` (same style as the existing video bubble), used both for live replies and when restoring conversation history.
- No `db.py` changes needed — this just reads the `image_url` / `images` field that's already on every product.

## 2. Manager gets a real email when an order is confirmed (`main.py`, `admin.html`)
- New `send_manager_email()` function sends an email via SMTP whenever `confirm_order_tool` fires — **in addition to** the existing dashboard notification (nothing about the dashboard changed).
- It also now emails on `request_human_tool` (a customer asking for a real person), since that's usually just as time-sensitive as an order.
- Hot-lead scoring notifications are left as dashboard-only on purpose — those can fire often and would spam an inbox.
- Controlled by a new **"Email me when an order is confirmed"** toggle in the admin Settings tab, wired to the `notifications_enabled` field your data model already had (it was defined but never used before).

### You need to add these to your `.env` file for email to actually send:
```
MANAGER_EMAIL=manager@yourstore.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-sending-address@gmail.com
SMTP_PASS=your-app-password
```
If you use Gmail: you can't use your normal password — you need a 16-character "App Password" from your Google Account's security settings (requires 2-Step Verification to be on). Any other SMTP provider (Outlook, Zoho, SendGrid, etc.) works the same way — same 4 variables, different host/port.

**If these aren't set, nothing breaks** — `send_manager_email()` just logs a warning and skips sending, so the bot keeps working normally while you set up email whenever you're ready.

## 3. Not changed
- `db.py` — untouched, because I still don't have it. These changes only ever *call* functions that `main.py` was already calling (`db.list_products()`, `db.create_notification()`, `db.get_business_settings()`), so they should work as-is against your real database. If your actual `db.py` stores `images`/`videos` differently than assumed (see `_first_image_url()` in `main.py`), paste it and I'll adjust that one function.
- Still just the **website widget** — WhatsApp/Telegram integration is a separate job. Say the word if you want that next.

## To deploy
Replace your existing `main.py` (or whatever your entry file is actually named — confirm it matches what your `Procfile`/start command points to), `app.js`, and `admin.html` with these three files, add the SMTP env vars, and redeploy.
