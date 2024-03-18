"use client";

import { useScreenDim } from "@/hooks/useScreenDim";
import React, { PropsWithChildren } from "react";
import { PixelGrid } from "./PixelGrid";

export const RootContent: React.FC<PropsWithChildren> = ({ children }) => {
  useScreenDim();

  return <PixelGrid>{children}</PixelGrid>;
};
