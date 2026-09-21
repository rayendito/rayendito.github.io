import rss from '@astrojs/rss';
import { AppConfig } from '@/utils/AppConfig';
import { getPosts } from '@/utils/posts';

export async function GET(context) {
	return rss({
		title: `${AppConfig.title} - blog`,
		description: AppConfig.description,
		site: context.site,
		items: getPosts().map((post) => ({ ...post.frontmatter, link: post.url })),
		stylesheet: './rss/styles.xsl',
		customData: `<language>en-us</language>`
	});
}
