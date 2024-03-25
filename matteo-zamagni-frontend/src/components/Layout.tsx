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
      {/* Header */}
      <div
        style={{
          gridRowStart: HEADER_OFFSET_Y + 1,
          gridRowEnd:
            HEADER_OFFSET_Y + 1 + HEADER_UPPER_HEIGHT + HEADER_LOWER_HEIGHT,
          gridColumnStart: gridDim.x / 2 - HEADER_CELL_WIDTH / 2 + 1,
          gridColumnEnd: gridDim.x / 2 + HEADER_CELL_WIDTH / 2 + 1,
        }}
      >
        {/* HEADER CELL */}
        <div
          className="w-full h-full grid"
          style={{
            gridTemplateColumns: `repeat(${HEADER_CELL_WIDTH}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${
              HEADER_UPPER_HEIGHT + HEADER_LOWER_HEIGHT
            }, minmax(0, 1fr))`,
          }}
        >
          <div
            className={"bg-red-500 row-span-full grid"}
            style={{
              gridColumnStart: 1,
              gridColumnEnd: 1 + SIDE_HEADER_CELL_WIDTH,
              gridRowStart: 1,
              gridRowEnd: HEADER_UPPER_HEIGHT + 1,
              gridTemplateColumns: `repeat(${SIDE_HEADER_CELL_WIDTH}, minmax(0, 1fr))`,
              gridTemplateRows: `repeat(${HEADER_UPPER_HEIGHT}, minmax(0, 1fr))`,
            }}
          >
            <Link
              href="/about"
              className="w-full bg-black col-span-full row-span-1 flex justify-center items-center"
            >
              about
            </Link>
          </div>
          <div
            className={`bg-green-500 row-span-full`}
            style={{
              gridColumnStart: 1 + SIDE_HEADER_CELL_WIDTH,
              gridColumnEnd:
                1 + SIDE_HEADER_CELL_WIDTH + CENTER_HEADER_CELL_WIDTH,
              gridRowStart: 1,
              gridRowEnd: HEADER_UPPER_HEIGHT + 1,
            }}
          >
            <Link
              href="/"
              className="w-full h-full flex justify-center items-center"
            >
              LOGO
            </Link>
          </div>
          <div
            className={`bg-blue-500 row-span-full`}
            style={{
              gridColumnStart: HEADER_CELL_WIDTH - SIDE_HEADER_CELL_WIDTH + 1,
              gridColumnEnd: HEADER_CELL_WIDTH + 1,
              gridRowStart: 1,
              gridRowEnd: HEADER_UPPER_HEIGHT + 1,
            }}
          >
            {/* <button className="w-full bg-black">index</button> */}
          </div>
          {/* HEADER LOWER */}
          <div
            className={`bg-yellow-500 row-span-full`}
            style={{
              gridColumnStart: 1,
              gridColumnEnd: HEADER_CELL_WIDTH + 1,
              gridRowStart: HEADER_UPPER_HEIGHT + HEADER_LOWER_HEIGHT - 1,
              gridRowEnd: HEADER_UPPER_HEIGHT + HEADER_LOWER_HEIGHT + 1,
            }}
          >
            {/* <button className="w-full bg-black">index</button> */}
          </div>
        </div>
      </div>
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
