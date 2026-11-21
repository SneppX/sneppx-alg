# Your progress

This page tracks how far you have explored the docs. Everything is stored in
your **browser's localStorage** — nothing is uploaded, there is no account, no
analytics and no leaderboard. Clearing site data resets it.

<div id="sx-progress-panel"></div>

## How it works

A page counts as read once you scroll to the bottom of it. The ring in the
right-hand sidebar shows overall completion, and it updates as you read.

## Suggested path

The tracks above are ordered. If you are new, follow **Start here → Autograd →
Tensors & ops → Scaling out → Practice**. Skipping ahead is fine; the bar just
fills in a different order.

## Reset

To start over, open your browser devtools and run:

```js
localStorage.removeItem("sx-progress-v1");
location.reload();
```

That is the whole "state" — a single JSON object in your own browser.
