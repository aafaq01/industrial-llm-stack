// Cross-browser check of the chat page. It EMPTIES the library of the rag-api it talks to, so point it at a scratch
// instance with its own index, never at your real one:
//   docker compose run -d --rm --name rag-scratch -p 127.0.0.1:8002:8000 -e ES_INDEX=scratch rag-api
//   npm i playwright && npx playwright install chromium firefox webkit
//   node check.mjs chromium firefox webkit        (BASE=http://127.0.0.1:8002 by default)
import { firefox, webkit, chromium } from "playwright";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const BASE = process.env.BASE || "http://127.0.0.1:8002";
const note = path.join(os.tmpdir(), "zephyr-notes.txt");
fs.writeFileSync(note, "Zephyr-9 pump handbook\n\nThe Zephyr-9 pump has a maximum operating pressure of 16 bar.\nIt must be serviced every 2,000 operating hours.\nThe seal kit part number is ZK-4471.\n");

const which = process.argv.slice(2);
const kinds = { firefox, webkit, chromium };
const results = [];

async function reset() {
  await fetch(`${BASE}/documents?confirm=yes`, { method: "DELETE" });
}

for (const name of which) {
  const steps = [];
  const ok = (label, cond, extra = "") => steps.push([label, !!cond, extra]);
  const errors = [];
  let browser;
  try {
    await reset();
    browser = await kinds[name].launch();
    const ctx = await browser.newContext({ viewport: { width: 1100, height: 760 } });
    const page = await ctx.newPage();
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push("pageerror: " + e.message));

    await page.goto(BASE + "/");
    await page.getByText("No documents yet").waitFor({ timeout: 10000 });
    ok("empty state shows the call to action", true);
    ok("question box disabled while empty", await page.locator("#q").isDisabled());

    await page.getByRole("button", { name: "Add documents" }).click();
    ok("panel opens", await page.locator("#docs").isVisible());
    await page.setInputFiles("#files", note);
    await page.getByText("Added, 1 passage").first().waitFor({ timeout: 60000 });
    ok("file uploaded and indexed", true);

    // a private web address must be refused and shown as an error
    await page.fill("#urls", "http://127.0.0.1:9200/");
    await page.click("#urls-add");
    await page.getByText("not on the public internet").waitFor({ timeout: 10000 });
    ok("private address refused with a reason", true);

    await page.keyboard.press("Escape");
    ok("Escape closes the panel", await page.locator("#docs").isHidden());
    ok("question box enabled after adding", await page.locator("#q").isEnabled());

    await page.fill("#q", "What is the maximum operating pressure of the Zephyr-9 pump?");
    await page.keyboard.press("Enter");
    await page.getByText("sources cited").first().waitFor({ timeout: 120000 });
    const answer = await page.locator(".msg:not(.user) .bubble").first().innerText();
    ok("answer streamed and mentions 16 bar", /16\s*bar/.test(answer), answer.slice(0, 80));
    await page.locator("button.cite").first().click();
    ok("citation chip opens and highlights the source", await page.locator(".src.used").first().isVisible());

    await page.click("#theme");
    const theme = await page.evaluate(() => document.documentElement.dataset.theme);
    ok("theme toggle works", theme === "dark" || theme === "light", String(theme));

    await page.reload();
    await page.locator(".msg:not(.user) .bubble").first().waitFor({ timeout: 10000 });
    ok("chat is restored after reload", true);

    await page.setViewportSize({ width: 390, height: 760 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
    ok("no sideways scrolling at phone width", !overflow);

    ok("no console errors", errors.length === 0, errors.join(" | ").slice(0, 200));
  } catch (e) {
    ok("run finished without an exception", false, String(e).split("\n")[0]);
  } finally {
    if (browser) await browser.close();
  }
  results.push([name, steps]);
}

let failed = 0;
for (const [name, steps] of results) {
  console.log(`\n== ${name} ==`);
  for (const [label, pass, extra] of steps) {
    if (!pass) failed++;
    console.log(`${pass ? "PASS" : "FAIL"}  ${label}${extra ? "  [" + extra + "]" : ""}`);
  }
}
process.exit(failed ? 1 : 0);
