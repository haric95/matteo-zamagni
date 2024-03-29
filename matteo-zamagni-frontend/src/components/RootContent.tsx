"use client";

import { useUpdateScreenDim } from "@/hooks/useScreenDim";
import React, { PropsWithChildren } from "react";
import { PixelGrid } from "./PixelGrid";
import { useGlobalContext } from "@/state/GlobalStore";
import { useSetDarkThemeClass } from "@/hooks/useSetDarkThemeClass";
import { Header } from "./Header";

const WaitForGridLoad: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim } = useGlobalContext();

  return gridDim ? <>{children}</> : null;
};

export const RootContent: React.FC<PropsWithChildren> = ({ children }) => {
  useUpdateScreenDim();
  useSetDarkThemeClass();

  return (
    <>
      <PixelGrid>
        <WaitForGridLoad>
          <Header />
          {children}
        </WaitForGridLoad>
      </PixelGrid>
    </>
  );
};
