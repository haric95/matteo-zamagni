import { RootContent } from "@/components/RootContent";
import { GlobalContextProvider } from "@/state/GlobalStore";
import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const ibmPlexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400"] });

export const metadata: Metadata = {
  title: "Matteo Zamagni",
  description: "Matteo Zamagni - Website",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={ibmPlexMono.className}>
        <GlobalContextProvider>
          <RootContent>{children}</RootContent>
        </GlobalContextProvider>
      </body>
    </html>
  );
}
