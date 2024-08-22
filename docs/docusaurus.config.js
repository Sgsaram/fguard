// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Forest Guard',
  tagline: '',
  favicon: 'img/logo.png',

  // Set the production url of your site here
  url: 'https://sgsaram.github.io/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/fguard/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'sgsaram', // Usually your GitHub org/user name.
  projectName: 'fguard-web', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
      },
      // Replace with your project's social card
      image: 'img/fguard-social-card.png',
      navbar: {
        title: 'Forest Guard',
        logo: {
          alt: 'Forest Guard Logo',
          src: 'img/logo.png',
        },
        // style: "dark",
        items: [




          // TODO



          // {
          //   label: 'About',
          //   to: '/blog/about',
          //   position: 'left'
          // },



          // TODO



          
          {
            label: 'Documentation',
            to: '/docs/forest-guard-cli/installation',
            position: 'left'
          },
          {
            label: 'Download',
            to: '/download',
            position: 'left'
          },
          // {
          //   type: 'docSidebar',
          //   sidebarId: 'tutorialSidebar',
          //   position: 'left',
          //   label: 'Tutorial',
          // },
          // {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://github.com/Sgsaram',
            html: `
              <div style="display: flex; justify-content: center; align-items: center;">
                <a href="https://github.com/Sgsaram" target="_blank" style="display: flex; justify-content: center; align-items: center; margin-right: 17px;">
                  <img src="/img/github-black.svg" width="25" height="25" />
                </a>
                <a href="https://t.me/sgsaram" target="_blank" style="display: flex; justify-content: center; align-items: center;">
                  <img src="/img/telegram-black.svg" width="25" height="25" />
                </a>
              </div>
            `,
            position: 'right',
          },
          // {
            // href: 'https://t.me/sgsaram',
            // html: `
            //   <a href="https://t.me/sgsaram" target="_blank" style="display: flex; justify-content: center; align-items: center; margin:">
            //     <img src="/img/telegram-black.svg" width="25" height="25" />
            //   </a>
            // `,
            // position: 'right',
          // },
        ],
      },
      footer: {
        style: 'dark',
        // links: [
        //   {
        //     title: 'Docs',
        //     items: [
        //       {
        //         label: 'Tutorial',
        //         to: '/docs/intro',
        //       },
        //     ],
        //   },
        //   {
        //     title: 'Community',
        //     items: [
        //       {
        //         label: 'Stack Overflow',
        //         href: 'https://stackoverflow.com/questions/tagged/docusaurus',
        //       },
        //       {
        //         label: 'Discord',
        //         href: 'https://discordapp.com/invite/docusaurus',
        //       },
        //       {
        //         label: 'Twitter',
        //         href: 'https://twitter.com/docusaurus',
        //       },
        //     ],
        //   },
        //   {
        //     title: 'More',
        //     items: [
        //       {
        //         label: 'Blog',
        //         to: '/blog',
        //       },
        //       {
        //         label: 'GitHub',
        //         href: 'https://github.com/facebook/docusaurus',
        //       },
        //     ],
        //   },
        // ],
        copyright: `${new Date().getFullYear()}, GPL-3.0 License. Made by Ivan Gronsky.`,
      },
      prism: {
        additionalLanguages: ['bash', 'toml'],
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
