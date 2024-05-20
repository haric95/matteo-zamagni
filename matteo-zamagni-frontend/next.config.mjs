// @ts-check

/**
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  /* config options here */
  images: {
    remotePatterns: [{ hostname: "placehold.co" }],
    formats: ["image/webp"],
  },
};

export default nextConfig;
