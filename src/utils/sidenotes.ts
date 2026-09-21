export function setupSidenotes() {
	const content = document.querySelector<HTMLElement>('.post-content');
	if (!content) return;
	const notes = [...content.querySelectorAll<HTMLElement>('[data-footnotes] > ol > li')];
	if (!notes.length) return;
	const references = [...content.querySelectorAll<HTMLAnchorElement>('[data-footnote-ref]')];
	const wide = window.matchMedia('(min-width: 800px)');

	function positionNotes() {
		if (!content) return;
		content.classList.toggle('has-sidenotes', wide.matches);
		content.style.minHeight = '';
		if (!wide.matches) return;

		const origin = content.getBoundingClientRect().top;
		let bottom = 0;
		for (const note of notes) {
			const reference = references.find(
				(link) => decodeURIComponent(link.hash.slice(1)) === note.id
			);
			if (!reference) continue;
			// Dense references stack in the margin instead of overlapping.
			const top = Math.max(reference.getBoundingClientRect().top - origin, bottom);
			note.style.top = `${top}px`;
			bottom = top + note.getBoundingClientRect().height + 16;
		}
		content.style.minHeight = `${Math.max(content.getBoundingClientRect().height, bottom)}px`;
	}

	let frame = 0;
	function scheduleLayout() {
		cancelAnimationFrame(frame);
		frame = requestAnimationFrame(positionNotes);
	}

	window.addEventListener('resize', scheduleLayout);
	content.addEventListener('load', scheduleLayout, true);
	void document.fonts.ready.then(scheduleLayout);
	positionNotes();
}
