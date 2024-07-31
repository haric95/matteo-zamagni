import { useGlobalContext } from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import Link from "next/link";
import React, { PropsWithChildren } from "react";
import { HeaderDateScroller } from "./HeaderDateScroller";
import { Logo } from "./Icons";
import { useIsMobile } from "@/hooks/useIsMobile";

const HEADER_CELL_WIDTH = 16;
export const HEADER_OFFSET_Y = 1;
export const HEADER_UPPER_HEIGHT = 4;
export const HEADER_LOWER_HEIGHT = 1;
export const TOTAL_HEADER_HEIGHT = HEADER_UPPER_HEIGHT + HEADER_LOWER_HEIGHT;
const SIDE_HEADER_CELL_WIDTH = Math.floor(HEADER_CELL_WIDTH / 3);
const CENTER_HEADER_CELL_WIDTH = HEADER_CELL_WIDTH - SIDE_HEADER_CELL_WIDTH * 2;

export const Header: React.FC<PropsWithChildren<PropsWithChildren>> = () => {
  const { gridDim } = useGlobalContext() as { gridDim: Dim2D };
  const isMobile = useIsMobile();

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
            className={"row-span-full grid"}
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
              className="w-full bg-background_Light dark:bg-background_Dark text-text_Light dark:text-offWhite transition-all duration-500 col-span-full row-span-1 flex justify-center items-center hover-glow hover:scale-105"
            >
              about
            </Link>
          </div>
          <div
            className={`row-span-full`}
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
              className="w-full h-full flex justify-center items-start translate-y-[-16px]"
            >
              <div className="bg-background_Light dark:bg-background_Dark transition-all duration-500">
                <Logo
                  height={96}
                  width={96}
                  className="hover:scale-105 transition-all duration-500 fill-black dark:fill-white"
                />
              </div>
            </Link>
          </div>
          <div
            className={`row-span-full`}
            style={{
              gridColumnStart: HEADER_CELL_WIDTH - SIDE_HEADER_CELL_WIDTH + 1,
              gridColumnEnd: HEADER_CELL_WIDTH + 1,
              gridRowStart: 1,
              gridRowEnd: HEADER_UPPER_HEIGHT + 1,
            }}
          >
            <Link
              href="/work-index"
              className="w-full bg-background_Light dark:bg-background_Dark text-text_Light dark:text-offWhite transition-all duration-500 col-span-full row-span-1 flex justify-center items-center hover-glow hover:scale-105"
            >
              index
            </Link>
          </div>
          {/* HEADER LOWER */}
          <div
            className={`row-span-full relative`}
            style={{
              gridColumnStart: isMobile ? 2 : 1,
              gridColumnEnd: isMobile
                ? HEADER_CELL_WIDTH
                : HEADER_CELL_WIDTH + 1,
              gridRowStart: HEADER_UPPER_HEIGHT + HEADER_OFFSET_Y,
              gridRowEnd:
                HEADER_UPPER_HEIGHT + HEADER_LOWER_HEIGHT + HEADER_OFFSET_Y,
            }}
          >
            <HeaderDateScroller loop />
          </div>
        </div>
      </div>
    </>
  );
};
