import { RootContent } from "@/components/RootContent";
import { GlobalContextProvider } from "@/state/GlobalStore";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

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
      <body className={inter.className}>
        <GlobalContextProvider>
          <RootContent>{children}</RootContent>
        </GlobalContextProvider>
      </body>
    </html>
  );
}
