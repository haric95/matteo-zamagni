"use client";

import { useUpdateScreenDim } from "@/hooks/useScreenDim";
import React, { PropsWithChildren, useCallback } from "react";
import { PixelGrid } from "./PixelGrid";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { useSetDarkThemeClass } from "@/hooks/useSetDarkThemeClass";
import { Header } from "./Header";
import { FooterLeft } from "./FooterLeft";
import PageAnimatePresence from "./PageAnimatePresence";
import { LoadingScreen } from "./LoadingScreen";
import { CreditsViewer } from "./CreditsViewer";

const WaitForGridLoad: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim } = useGlobalContext();

  return gridDim ? <>{children}</> : null;
};

export const RootContent: React.FC<PropsWithChildren> = ({ children }) => {
  useUpdateScreenDim();
  useSetDarkThemeClass();

  const { creditsIsOpen } = useGlobalContext();
  const dispatch = useGlobalContextDispatch();

  const handleCloseCredits = useCallback(() => {
    if (dispatch) {
      dispatch({ type: "CLOSE_CREDITS" });
    }
  }, [dispatch]);

  return (
    <>
      <LoadingScreen>
        <PixelGrid>
          <WaitForGridLoad>
            <Header />
            <PageAnimatePresence>
              {children}
              {creditsIsOpen && (
                <CreditsViewer handleClose={handleCloseCredits} />
              )}
            </PageAnimatePresence>
            <FooterLeft />
          </WaitForGridLoad>
        </PixelGrid>
      </LoadingScreen>
    </>
  );
};
