"use client";

import { useUpdateScreenDim } from "@/hooks/useScreenDim";
import React, { PropsWithChildren } from "react";
import { PixelGrid } from "./PixelGrid";

export const RootContent: React.FC<PropsWithChildren> = ({ children }) => {
  useUpdateScreenDim();

  return (
    <>
      <PixelGrid></PixelGrid>
      <div className="w-full h-full relative">{children}</div>
    </>
  );
};
