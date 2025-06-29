/** @type {import('next').NextConfig} */
const nextConfig = {

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