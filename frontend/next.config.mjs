/** @type {import('next').NextConfig} */
const nextConfig = {
  // Cloudfront 등 외부 도메인 허용 (CLAUDE.md 요구사항)
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [{ key: "X-Frame-Options", value: "SAMEORIGIN" }],
      },
    ];
  },
};

export default nextConfig;
