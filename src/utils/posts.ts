import type { MarkdownInstance } from 'astro';

export function getPosts() {
	const modules = import.meta.glob<MarkdownInstance<Record<string, any>>>('../posts/*.md', {
		eager: true
	});

	return Object.entries(modules)
		.filter(([, post]) => import.meta.env.DEV || post.frontmatter.draft !== true)
		.map(([path, post]) => {
			const slug = path.split('/').pop()!.replace(/\.md$/, '');
			return { ...post, slug, url: `/posts/${slug}/` };
		});
}
