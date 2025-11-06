const path = require('path');
/** @type {import('next').NextConfig} */
const withProxyRewrites = () => {
  const target = process.env.BACKEND_ORIGIN || process.env.NEXT_PUBLIC_BACKEND_URL;
  if (!target) {
    return [];
  }
  const normalised = target.replace(/\/$/, '');
  return [
    {
      source: '/edge/:path*',
      destination: `${normalised}/:path*`,
    },
    {
      source: '/alerts/:path*',
      destination: `${normalised}/alerts/:path*`,
    },
  ];
};

const nextConfig = {
  reactStrictMode: true,
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      net: false,
      tls: false,
    };
    return config;
  },
  async rewrites() {
    return withProxyRewrites();
  },
};

module.exports = nextConfig;
