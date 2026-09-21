const opener = '\\footnote{';

// Register an inline Markdown construct so code blocks and math remain untouched.
function tokenizeFootnote(effects, ok, nok) {
	let index = 0;
	let depth = 1;
	let escaped = false;
	let codeFence = 0;
	let ticks = 0;
	return start;

	function start(code) {
		if (index === 0) effects.enter('inlineFootnote');
		if (code !== opener.charCodeAt(index)) return nok(code);
		effects.consume(code);
		index++;
		return index === opener.length ? body : start;
	}

	function body(code) {
		if (code === null || code === -5 || code === -4 || code === -3) return nok(code);
		if (escaped) {
			escaped = false;
		} else if (code === 96) {
			ticks = 1;
			effects.consume(code);
			return backticks;
		} else if (!codeFence) {
			if (code === 92) escaped = true;
			if (code === 123) depth++;
			if (code === 125 && --depth === 0) {
				effects.consume(code);
				effects.exit('inlineFootnote');
				return ok;
			}
		}
		effects.consume(code);
		return body;
	}

	function backticks(code) {
		if (code === 96) {
			ticks++;
			effects.consume(code);
			return backticks;
		}
		if (!codeFence) codeFence = ticks;
		else if (codeFence === ticks) codeFence = 0;
		return body(code);
	}
}

export function remarkInlineFootnotes() {
	const processor = this;
	const data = processor.data();
	(data.micromarkExtensions ??= []).push({
		text: { 92: { name: 'inlineFootnote', tokenize: tokenizeFootnote } }
	});
	(data.fromMarkdownExtensions ??= []).push({
		enter: {
			inlineFootnote(token) {
				this.enter({ type: 'inlineFootnote', value: '' }, token);
			}
		},
		exit: {
			inlineFootnote(token) {
				const node = this.stack[this.stack.length - 1];
				node.value = this.sliceSerialize(token).slice(opener.length, -1);
				this.exit(token);
			}
		}
	});

	return (tree) => {
		const identifiers = new Set();
		const definitions = [];
		let number = 0;
		function collect(node) {
			if (node.identifier) identifiers.add(node.identifier.toLowerCase());
			node.children?.forEach(collect);
		}
		collect(tree);

		function transform(parent) {
			if (!parent.children) return;
			parent.children = parent.children.map((node) => {
				if (node.type !== 'inlineFootnote') {
					transform(node);
					return node;
				}
				const note = processor.parse(node.value);
				collect(note);
				let identifier;
				do identifier = `inline-note-${++number}`;
				while (identifiers.has(identifier));
				identifiers.add(identifier);
				transform(note);
				definitions.push({ type: 'footnoteDefinition', identifier, children: note.children });
				return { type: 'footnoteReference', identifier };
			});
		}
		transform(tree);
		tree.children.push(...definitions);
	};
}
