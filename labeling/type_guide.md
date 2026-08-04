# Type guide — keep this open while labelling

These are the **same rules the model is prompted with** (`scam-type-classification/prompt.yaml`,
`type_guide`). Apply them literally, even where your instinct disagrees.

That matters more than it sounds. "Is an Amazon account alert `tech support` or `bank
alert`?" has no true answer — it is a definitional choice, and `prompt.yaml` already made
it. If you label by gut you will drift from that rule and score the model down for
obeying its own spec. The question this eval can answer is *does the model follow the
stated rules*. If a rule is wrong, change it in `prompt.yaml` and re-run the model —
don't work around it in the labels.

## Standing premise

Every message here has **already been flagged as suspicious** by stage 1. Do not
re-judge whether it is a scam — decide which **kind** it is.

In particular: a message that reads as harmless small talk is **not** `other`. A friendly
stranger, a wrong number, or a golf/dinner memory that never happened is the opening move
of a **romance** scam; the ask comes in a later message you cannot see.

## The 13 types

- **government impersonation** — claims to be a tax, benefits, legal, police or
  motor-vehicle authority. Threatens fines, warrants, suspended benefits or a court case.
  *e.g. "IRS FINAL NOTICE: a warrant is issued for your arrest. Call now to settle."*

- **tech support** — claims an account, device or subscription has a problem and needs you
  to log in, call support, or pay to restore it. Any account alert from a shop, streaming
  service, phone carrier, social network or tech company (Amazon, Apple, Netflix, Facebook,
  T-Mobile) belongs here — including a locked account, an unauthorised purchase or a
  payment. **A charge being mentioned does not make it `bank alert`; the sender being a
  bank does.**
  *e.g. "Your Netflix membership is on hold. Update your billing: netflx-billing.top"*
  *e.g. "Your Amazon account is locked after an unauthorised $999 order. Verify now."*

- **bank alert** — claims to be your bank or a payment app about a transaction, a locked
  card or suspicious activity. **Only** banks and payment apps (Wells Fargo, Chase, PayPal,
  Venmo, Cash App) — an Amazon or Apple account alert is `tech support`.
  *e.g. "Wells Fargo: Did you authorise $482.19 to AMZN? Reply NO to dispute."*

- **delivery and toll** — anything about a parcel, courier, customs charge or road toll:
  held packages, redelivery, an unpaid shipping or customs fee. Any unpaid fee owed to a
  courier (USPS, UPS, FedEx, DHL) belongs here, **not** under utilities.
  *e.g. "USPS: your package is held, $0.35 customs fee due: usps-redeliver.icu"*

- **family emergency** — pretends to be a relative in trouble and needing money now.
  *e.g. "Grandma it's me, I'm in jail and can't talk. Please send bail money."*

- **romance** — a stranger opening a conversation: a wrong number, a misremembered name,
  "long time no see", a shared meal or golf game that never happened. There is usually
  **no link, no money request and nothing threatening** in this first message; that is what
  makes it this category rather than `other`.
  *e.g. "Hi! Is this still Amy's number? I'm Lisa, we met at the yoga retreat :)"*

- **investment and crypto** — promises trading profits, crypto giveaways or an investment tip.
  *e.g. "BTC signal group: our members made 340% last week. Join free."*

- **prize and lottery** — says you have won something, or are selected for a reward or gift card.
  *e.g. "CONGRATS! You have been selected for a $500 Costco gift card. Claim here."*

- **charity** — asks for a donation, often tied to a disaster or a veterans' cause.
  *e.g. "Help the hurricane relief fund - any amount helps. Donate: give-relief.xyz"*

- **Medicare and health** — Medicare, insurance, prescriptions, braces, test kits or medical
  devices, usually "free" and needing your details.
  *e.g. "New Medicare benefit: free back brace approved. Confirm your ID to ship."*

- **utility shutoff** — threatens to cut off power, water, gas or internet over an unpaid
  bill. **Only household utilities** — a courier's unpaid fee is `delivery and toll`.
  *e.g. "FINAL WARNING: your electricity will be disconnected today. Pay now."*

- **job offer** — offers easy remote work, high pay for little effort, or a hiring interview
  you never applied for.
  *e.g. "We reviewed your resume. $35/hr remote data entry, start today. Text YES."*

- **other** — a suspicious message that genuinely fits none of the types above. Use this
  sparingly; re-read the romance rule before reaching for it.

## The known-hard boundaries

Three collisions produce most of the disagreement. Decide them the same way every time:

| Looks like | Rule |
|---|---|
| Tax refund / IRS with a money angle | `government impersonation` — the impersonated *authority* wins over the fact that money is involved. Not `investment and crypto`. |
| Amazon / Netflix / Apple charge or locked account | `tech support`. The sender is not a bank. |
| Unpaid USPS / DHL fee | `delivery and toll`. Not `utility shutoff`, not `bank alert`. |

## Filling in the columns

- **`label`** — one type name from the list above, copied exactly. Always required.
- **`second_label`** — a genuinely defensible alternative, or blank. Use it. An IRS refund
  phish really is both an authority impersonation and a financial lure; forcing one answer
  puts a ceiling on measurable accuracy that is not the model's fault. Scoring runs both
  strict (primary only) and lenient (either counts), and the gap between them measures the
  taxonomy's ambiguity rather than the model's error.
- **`confidence`** — `high` / `med` / `low`. Your own certainty, not the message's.
- **`flag`** — leave blank normally. Set `NOT_A_SCAM` if the message is plainly legitimate,
  `TRUNCATED` if the text is cut off or garbled. SmishTank is scraped from screenshots and
  some rows are junk; flagged rows are excluded from scoring rather than counted against
  anyone. Rows pre-filled with `NEEDS_SOURCING` are empty work-orders — see README.
- **`note`** — free text, only when something is worth remembering.
