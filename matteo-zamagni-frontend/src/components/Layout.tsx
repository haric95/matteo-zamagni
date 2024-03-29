import { useGlobalContext } from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import Link from "next/link";
import React, { ReactElement } from "react";
import { PropsWithChildren } from "react";

const HEADER_OFFSET_Y = 1;
const HEADER_CELL_WIDTH = 16;
const HEADER_UPPER_HEIGHT = 4;
const HEADER_LOWER_HEIGHT = 2;
const SIDE_HEADER_CELL_WIDTH = Math.floor(HEADER_CELL_WIDTH / 3);
const CENTER_HEADER_CELL_WIDTH = HEADER_CELL_WIDTH - SIDE_HEADER_CELL_WIDTH * 2;

const FOOTER_LEFT_HEIGHT = 3;
const FOOTER_LEFT_WIDTH = 1;
const FOOTER_LEFT_OFFSET_X = 2;
const FOOTER_LEFT_OFFSET_Y = 2;

const FOOTER_RIGHT_OFFSET_X = 2;
const FOOTER_RIGHT_OFFSET_Y = 2;

type LayoutProps = {
  footerRightHeight?: number;
  footerRightWidth?: number;
  footerRightComponent?: ReactElement;
};

export const Layout: React.FC<PropsWithChildren<LayoutProps>> = ({
  children,
  footerRightHeight,
  footerRightWidth,
  footerRightComponent,
}) => {
  const { gridDim } = useGlobalContext() as { gridDim: Dim2D };

  return (
    <>
      {/* Footer Left */}
      <div
        className="grid bg-background_Light dark:bg-background_Dark"
        style={{
          gridColumnStart: FOOTER_LEFT_OFFSET_X + 1,
          gridColumnEnd: FOOTER_LEFT_OFFSET_X + 1 + FOOTER_LEFT_WIDTH,
          gridRowStart:
            gridDim.y - (FOOTER_LEFT_HEIGHT + (FOOTER_LEFT_OFFSET_Y - 1)),
          gridRowEnd:
            gridDim.y -
            (FOOTER_LEFT_HEIGHT + (FOOTER_LEFT_OFFSET_Y - 1)) +
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
        <div className="row-span-1 flex justify-center items-center text-[24px]">
          ©
        </div>
      </div>
      {/* Footer Right */}
      {footerRightComponent && footerRightWidth && footerRightHeight && (
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
          {footerRightComponent}
        </div>
      )}
      {/* CONTENT */}
      {/* <div
        className=""
        style={{
          gridColumnStart:
            gridDim.x - (footerRightWidth + FOOTER_RIGHT_OFFSET_X),
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
          // gridTemplateColumns: `repeat(${FOOTER_RIGHT_WIDTH}, minmax(0, 1fr))`,
          // gridTemplateRows: `repeat(${FOOTER_RIGHT_HEIGHT}, minmax(0, 1fr))`,
        }}
      ></div> */}
      {children}
    </>
  );
};
