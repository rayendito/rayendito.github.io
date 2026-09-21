import assert from 'node:assert/strict';
import test from 'node:test';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkInlineFootnotes } from '../src/utils/remarkInlineFootnotes.mjs';

const processor = await createMarkdownProcessor({
	syntaxHighlight: false,
	remarkPlugins: [remarkInlineFootnotes, remarkMath],
	rehypePlugins: [rehypeKatex]
});
const render = async (text) => (await processor.render(text)).code;
const references = (html) => (html.match(/data-footnote-ref=/g) ?? []).length;

test('inline notes use standard footnotes and parse Markdown formatting', async () => {
	const html = await render(
		String.raw`Picking the most likely\footnote{This is **naive**; see [why](https://example.com).} token.`
	);
	assert.equal(references(html), 1);
	assert.match(html, /<strong>naive<\/strong>/);
	assert.match(html, /href="https:\/\/example.com"/);
	assert.match(html, /data-footnote-backref/);
	assert.doesNotMatch(html, /\\footnote/);
});

test('balanced math braces, escaped braces, and braces inside code work', async () => {
	const html = await render(
		'Text\\footnote{Math $p_{\\theta}(x)$, literal \\{brace\\}, code `}` and ``a`{b``.} After.'
	);
	assert.equal(references(html), 1);
	assert.match(html, /class="katex"/);
	assert.match(html, /literal \{brace\}/);
	assert.match(html, /<code>}<\/code>/);
	assert.match(html, /After\./);
});

test('code examples, escaped commands, and math do not create notes', async () => {
	for (const text of [
		'`\\footnote{example}`',
		'```tex\n\\footnote{example}\n```',
		String.raw`\\footnote{example}`,
		String.raw`$\text{\footnote{example}}$`
	])
		assert.equal(references(await render(text)), 0);
});

test('unclosed commands stay literal', async () => {
	const html = await render(String.raw`Text\footnote{not finished`);
	assert.equal(references(html), 0);
	assert.match(html, /\\footnote\{not finished/);
});

test('inline and named notes coexist without identifier collisions', async () => {
	const html = await render(
		'A[^inline-note-1] B\\footnote{Second note} C\\footnote{Third note}\n\n[^inline-note-1]: First note'
	);
	assert.equal(references(html), 3);
	const ids = [...html.matchAll(/<li id="([^"]+)"/g)].map((match) => match[1]);
	assert.equal(new Set(ids).size, 3);
	for (const label of ['First note', 'Second note', 'Third note'])
		assert.match(html, new RegExp(label));
});

test('notes render repeatedly without leaking state', async () => {
	const source = 'A\\footnote{A short note}';
	assert.equal(references(await render(source)), 1);
	assert.equal(await render(source), await render(source));
});

test('a command interrupted by a newline remains literal', async () => {
	assert.equal(references(await render('A\\footnote{First line\nsecond line}')), 0);
});
