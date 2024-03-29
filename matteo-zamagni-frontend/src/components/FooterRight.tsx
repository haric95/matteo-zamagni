import { useGlobalContext } from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import React, { PropsWithChildren, ReactElement } from "react";

const FOOTER_RIGHT_OFFSET_X = 2;
const FOOTER_RIGHT_OFFSET_Y = 2;

type FooterRightProps = PropsWithChildren<{
  footerRightHeight?: number;
  footerRightWidth?: number;
}>;

export const FooterRight: React.FC<PropsWithChildren<FooterRightProps>> = ({
  footerRightHeight,
  footerRightWidth,
  children,
}) => {
  const { gridDim } = useGlobalContext() as { gridDim: Dim2D };

  return (
    <>
      {/* Footer Right */}
      {children && footerRightWidth && footerRightHeight && (
        <div
          className="grid bg-background_Light dark:bg-background_Dark"
          style={{
            gridColumnStart:
              gridDim.x - (footerRightWidth + FOOTER_RIGHT_OFFSET_X - 1),
            gridColumnEnd:
              gridDim.x -
              (footerRightWidth + FOOTER_RIGHT_OFFSET_X - 1) +
              footerRightWidth,
            gridRowStart:
              gridDim.y - (footerRightHeight + (FOOTER_RIGHT_OFFSET_Y - 1)),
            gridRowEnd:
              gridDim.y -
              (footerRightHeight + (FOOTER_RIGHT_OFFSET_Y - 1)) +
              footerRightHeight,
            gridTemplateColumns: `repeat(${footerRightWidth}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${footerRightHeight}, minmax(0, 1fr))`,
          }}
        >
          {children}
        </div>
      )}
    </>
  );
};
