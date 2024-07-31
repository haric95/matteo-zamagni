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
  onMobileFooterMenuClose?: () => void;
  breakpointSize?: number;
}>;

export const FooterRight: React.FC<PropsWithChildren<FooterRightProps>> = ({
  footerRightHeight,
  footerRightWidth,
  isMounted = true,
  mobileTitleComponent,
  onMobileFooterMenuClose,
  breakpointSize,
  children,
}) => {
  const { gridDim, mobileFooterMenuOpen } = useGlobalContext() as {
    gridDim: Dim2D;
    mobileFooterMenuOpen: boolean;
  };
  const dispatch = useGlobalContextDispatch();
  const isMobile = useIsMobile(breakpointSize);

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
              className={`fixed w-screen h-[calc(100dvh)] top-0 left-0 bg-black z-[999] transition-all duration-500 text-sm md:text-md ${
                mobileFooterMenuOpen
                  ? "opacity-90"
                  : "opacity-0 pointer-events-none"
              }`}
              onClick={() => {
                if (dispatch) {
                  dispatch({ type: "SET_MOBILE_FOOTER_MENU", isOpen: false });
                  if (onMobileFooterMenuClose) {
                    onMobileFooterMenuClose();
                  }
                }
              }}
            >
              {mobileFooterMenuOpen && (
                <GridChild
                  x={gridDim.x - 1 - footerRightWidth}
                  y={gridDim.y - 1 - footerRightHeight}
                  width={footerRightWidth}
                  height={footerRightHeight}
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                >
                  {children}
                </GridChild>
              )}
            </GridChild>
            <MotionGridChild
              {...DEFAULT_ANIMATE_MODE}
              height={1}
              width={15}
              x={gridDim.x - 16}
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
            className="grid"
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
