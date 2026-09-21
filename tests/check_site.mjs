import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, stat, mkdir, readdir } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '../site/node_modules/playwright-core/index.mjs';

const root = resolve(fileURLToPath(new URL('../site/.vitepress/dist/', import.meta.url)));
const output = resolve(fileURLToPath(new URL('../outputs/site-check/', import.meta.url)));
await mkdir(output, { recursive: true });
const base = '/hvac-ai-pid-demo/';
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.svg': 'image/svg+xml', '.json': 'application/json', '.woff2': 'font/woff2', '.woff': 'font/woff' };
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (!pathname.startsWith(base)) { res.writeHead(404).end(); return; }
    let file = resolve(root, pathname.slice(base.length) || 'index.html');
    if (file !== root && !file.startsWith(root + sep)) { res.writeHead(403).end(); return; }
    if (!extname(file)) file += '.html';
    if ((await stat(file)).isDirectory()) file = resolve(file, 'index.html');
    res.writeHead(200, { 'Content-Type': types[extname(file)] || 'application/octet-stream' });
    res.end(await readFile(file));
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const chapterDir = fileURLToPath(new URL('../site/chapters/', import.meta.url));
const slugs = (await readdir(chapterDir)).filter(name => name.endsWith('.md')).map(name => name.slice(0, -3));
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const slug of ['', ...slugs.map(s => `chapters/${s}`)]) {
    const response = await page.goto(origin + base + slug, { waitUntil: 'networkidle' });
    assert.equal(response.status(), 200);
    await page.locator('.vp-doc h1').waitFor();
    assert.equal(await page.locator('.katex-error').count(), 0, `${slug}: formula error`);
    const images = await page.locator('.vp-doc img').evaluateAll(images => images.map(img => ({
      src: img.src, complete: img.complete, width: img.naturalWidth,
    })));
    assert.ok(images.every(image => image.complete && image.width > 0), `${slug}: broken image`);
    if (slug && !slug.includes('08-llm') && !slug.includes('09-comparison')) {
      assert.ok(await page.locator('.katex').count() > 0, `${slug}: formulas missing`);
    }
    const links = await page.locator('a[href]').evaluateAll(elements => elements.map(a => a.href));
    for (const href of links.filter(link => link.startsWith(origin + '/'))) {
      assert.ok(href.startsWith(origin + base), `Internal link escaped the Pages base: ${href}`);
    }
    for (const href of new Set(links.filter(link => link.startsWith(origin + base)))) {
      const check = await page.request.get(href.split('#')[0]);
      assert.equal(check.status(), 200, `Broken internal link: ${href}`);
      const fragment = decodeURIComponent(new URL(href).hash.slice(1));
      if (fragment && (check.headers()['content-type'] || '').includes('text/html')) {
        const html = await check.text();
        assert.ok(html.includes(`id="${fragment}"`) || html.includes(`id='${fragment}'`),
          `Broken internal fragment: ${href}`);
      }
    }
    await page.screenshot({ path: resolve(output, (slug.split('/').pop() || 'index') + '.png'), fullPage: true });
    if (slug === 'chapters/14-pg4pi') {
      await page.screenshot({ path: resolve(output, 'desktop-pg4pi-intro.png') });
    }
  }
  await page.goto(origin + base + 'chapters/06-qlearning');
  assert.ok((await page.locator('.vp-doc').innerText()).includes('update_q'));
  await page.setViewportSize({ width: 390, height: 844 });
  for (const slug of slugs) {
    await page.goto(origin + base + 'chapters/' + slug, { waitUntil: 'networkidle' });
    const dimensions = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth }));
    assert.ok(dimensions.document <= dimensions.viewport + 1, `${slug}: mobile page overflows horizontally`);
    await page.screenshot({ path: resolve(output, 'mobile-' + slug + '.png'), fullPage: true });
    if (slug === '13-crossq') {
      await page.screenshot({ path: resolve(output, 'mobile-crossq-intro.png') });
    }
  }
  assert.deepEqual(errors, []);
  console.log(`Checked ${slugs.length + 1} pages: formulas, images, source snippets, links, mobile layout.`);
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
