/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['@alloc/quick-lru'],
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false, // avoid fs crash in browser
    };
    return config;
  },
  experimental: {
    serverActions: {
      allowedOrigins: ['localhost:3000', '127.0.0.1:3000','https://mira-aitutor.netlify.app'],
    },
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { unoptimized: true },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://mira-backend-fcdndhgegjdhghf2.centralindia-01.azurewebsites.net/:path*',
      },
    ];
  },
};

module.exports = nextConfig;