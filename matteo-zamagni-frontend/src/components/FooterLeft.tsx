import { useIsMobile } from "@/hooks/useIsMobile";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import React, { PropsWithChildren, useCallback, useId } from "react";

const FOOTER_LEFT_HEIGHT = 3;
const FOOTER_LEFT_WIDTH = 1;
const FOOTER_LEFT_OFFSET_X = 2;
const FOOTER_LEFT_OFFSET_Y = 2;
const FOOTER_LEFT_OFFSET_X_MOBILE = 1;
const FOOTER_LEFT_OFFSET_Y_MOBILE = 1;

export const FooterLeft: React.FC<PropsWithChildren> = () => {
  const { gridDim } = useGlobalContext() as { gridDim: Dim2D };
  const dispatch = useGlobalContextDispatch();
  const isMobile = useIsMobile();

  const handleOpenCredits = useCallback(() => {
    if (dispatch) {
      dispatch({ type: "OPEN_CREDITS" });
    }
  }, [dispatch]);

  const offsetX = isMobile ? FOOTER_LEFT_OFFSET_X_MOBILE : FOOTER_LEFT_OFFSET_X;
  const offsetY = isMobile ? FOOTER_LEFT_OFFSET_Y_MOBILE : FOOTER_LEFT_OFFSET_Y;

  return (
    <>
      {/* Footer Left */}
      <footer
        className="grid bg-background_Light dark:bg-background_Dark transition-all duration-500 z-10"
        style={{
          gridColumnStart: offsetX + 1,
          gridColumnEnd: offsetX + 1 + FOOTER_LEFT_WIDTH,
          gridRowStart: gridDim.y - (FOOTER_LEFT_HEIGHT + (offsetY - 1)),
          gridRowEnd:
            gridDim.y -
            (FOOTER_LEFT_HEIGHT + (offsetY - 1)) +
            FOOTER_LEFT_HEIGHT,
          gridTemplateColumns: `repeat(${FOOTER_LEFT_WIDTH}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${FOOTER_LEFT_HEIGHT}, minmax(0, 1fr))`,
        }}
      >
        <a
          href="https://www.instagram.com/matteo.zamagni/"
          target="_blank"
          rel="noreferrer"
          className="row-span-1 flex justify-center items-center"
        >
          IG
        </a>
        <a
          href="https://vimeo.com/matteozamagni"
          target="_blank"
          rel="noreferrer"
          className="row-span-1 flex justify-center items-center"
        >
          VM
        </a>
        <button
          onClick={handleOpenCredits}
          className="row-span-1 flex justify-center items-center text-[24px]"
        >
          ©
        </button>
      </footer>
    </>
  );
};
