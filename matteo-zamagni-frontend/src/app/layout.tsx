import { RootContent } from "@/components/RootContent";
import { GlobalContextProvider, useGlobalContext } from "@/state/GlobalStore";
import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import localFont from "next/font/local";
import "./globals.css";
import { ErrorBoundary } from "next/dist/client/components/error-boundary";
import { ErrorComponent } from "@/components/ErrorComponent";

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-mono",
});
export const myFont = localFont({
  src: "../fonts/Neutronica Geometric.ttf",
  variable: "--font-decoration",
});

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
      <body
        className={`${ibmPlexMono.variable} ${myFont.variable} font-mono bg-black selection:bg-highlight selection:text-black overflow-hidden`}
      >
        <GlobalContextProvider>
          <ErrorBoundary errorComponent={ErrorComponent}>
            <RootContent>{children}</RootContent>
          </ErrorBoundary>
        </GlobalContextProvider>
      </body>
    </html>
  );
}
