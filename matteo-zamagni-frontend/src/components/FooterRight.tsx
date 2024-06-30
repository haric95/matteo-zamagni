import { DEFAULT_ANIMATE_MODE } from "@/const";
import { useIsMobile } from "@/hooks/useIsMobile";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import React, { PropsWithChildren } from "react";
import { GridChild } from "./GridChild";
import { MotionGridChild } from "./MotionGridChild";

const FOOTER_RIGHT_OFFSET_X = 2;
const FOOTER_RIGHT_OFFSET_Y = 2;
const FOOTER_RIGHT_OFFSET_X_MOBILE = 1;
const FOOTER_RIGHT_OFFSET_Y_MOBILE = 1;

type FooterRightProps = PropsWithChildren<{
  footerRightHeight?: number;
  footerRightWidth?: number;
  isMounted?: boolean;
  mobileTitleComponent?: JSX.Element;
}>;

export const FooterRight: React.FC<PropsWithChildren<FooterRightProps>> = ({
  footerRightHeight,
  footerRightWidth,
  isMounted = true,
  mobileTitleComponent,
  children,
}) => {
  const { gridDim, mobileFooterMenuOpen } = useGlobalContext() as {
    gridDim: Dim2D;
    mobileFooterMenuOpen: boolean;
  };
  const dispatch = useGlobalContextDispatch();
  const isMobile = useIsMobile();

  const handleMobileFooterMenuOpen = () => {
    if (dispatch) {
      dispatch({ type: "SET_MOBILE_FOOTER_MENU", isOpen: true });
    }
  };

  const offsetX = isMobile
    ? FOOTER_RIGHT_OFFSET_X_MOBILE
    : FOOTER_RIGHT_OFFSET_X;
  const offsetY = isMobile
    ? FOOTER_RIGHT_OFFSET_Y_MOBILE
    : FOOTER_RIGHT_OFFSET_Y;

  return (
    <AnimatePresence>
      {/* Footer Right */}
      {children &&
        footerRightWidth &&
        footerRightHeight &&
        isMounted &&
        (isMobile ? (
          <>
            <GridChild
              x={0}
              y={0}
              width={gridDim.x}
              height={gridDim.y}
              className={`fixed w-screen h-screen bg-black z-[999] transition-all duration-500 ${
                mobileFooterMenuOpen
                  ? "opacity-90"
                  : "opacity-0 pointer-events-none"
              }`}
            >
              {mobileFooterMenuOpen && (
                <GridChild
                  x={gridDim.x - 1 - footerRightWidth}
                  y={gridDim.y - 1 - footerRightHeight}
                  width={footerRightWidth}
                  height={footerRightHeight}
                >
                  {children}
                </GridChild>
              )}
            </GridChild>
            <MotionGridChild
              {...DEFAULT_ANIMATE_MODE}
              height={1}
              width={8}
              x={gridDim.x - 9}
              y={gridDim.y - 2}
              isGrid={false}
              className=""
            >
              <button
                onClick={() => {
                  handleMobileFooterMenuOpen();
                }}
                className="icon-hover-glow w-full h-full text-right flex items-center justify-end"
              >
                {mobileTitleComponent}
              </button>
            </MotionGridChild>
          </>
        ) : (
          <motion.div
            {...DEFAULT_ANIMATE_MODE}
            className="grid bg-background_Light dark:bg-background_Dark"
            style={{
              gridColumnStart: gridDim.x - (footerRightWidth + offsetX - 1),
              gridColumnEnd:
                gridDim.x - (footerRightWidth + offsetX - 1) + footerRightWidth,
              gridRowStart: gridDim.y - (footerRightHeight + (offsetY - 1)),
              gridRowEnd:
                gridDim.y -
                (footerRightHeight + (offsetY - 1)) +
                footerRightHeight,
              gridTemplateColumns: `repeat(${footerRightWidth}, minmax(0, 1fr))`,
              gridTemplateRows: `repeat(${footerRightHeight}, minmax(0, 1fr))`,
            }}
          >
            {children}
          </motion.div>
        ))}
    </AnimatePresence>
  );
};
