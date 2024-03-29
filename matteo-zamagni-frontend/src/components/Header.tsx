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

type LayoutProps = {
  footerRightHeight?: number;
  footerRightWidth?: number;
  footerRightComponent?: ReactElement;
};

export const Header: React.FC<PropsWithChildren<LayoutProps>> = ({
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
        className="z-10"
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
    </>
  );
};
