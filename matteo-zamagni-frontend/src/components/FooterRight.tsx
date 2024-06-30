import { DEFAULT_ANIMATE_MODE } from "@/const";
import { useIsMobile } from "@/hooks/useIsMobile";
import { useGlobalContext } from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import React, { PropsWithChildren, ReactElement } from "react";

const FOOTER_RIGHT_OFFSET_X = 2;
const FOOTER_RIGHT_OFFSET_Y = 2;
const FOOTER_RIGHT_OFFSET_X_MOBILE = 1;
const FOOTER_RIGHT_OFFSET_Y_MOBILE = 1;

type FooterRightProps = PropsWithChildren<{
  footerRightHeight?: number;
  footerRightWidth?: number;
  isMounted?: boolean;
}>;

export const FooterRight: React.FC<PropsWithChildren<FooterRightProps>> = ({
  footerRightHeight,
  footerRightWidth,
  isMounted = true,
  children,
}) => {
  const { gridDim } = useGlobalContext() as { gridDim: Dim2D };
  const isMobile = useIsMobile();

  const offsetX = isMobile
    ? FOOTER_RIGHT_OFFSET_X_MOBILE
    : FOOTER_RIGHT_OFFSET_X;
  const offsetY = isMobile
    ? FOOTER_RIGHT_OFFSET_Y_MOBILE
    : FOOTER_RIGHT_OFFSET_Y;

  return (
    <AnimatePresence>
      {/* Footer Right */}
      {children && footerRightWidth && footerRightHeight && isMounted && (
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
      )}
    </AnimatePresence>
  );
};
