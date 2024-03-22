"use client";
import { useGlobalContext } from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";

const HEADER_CELL_HEIGHT = 6;
const HEADER_OFFSET_Y = 2;
const HEADER_CELL_WIDTH = 16;
const SIDE_HEADER_CELL_WIDTH = Math.floor(HEADER_CELL_WIDTH / 3);
const CENTER_HEADER_CELL_WIDTH = HEADER_CELL_WIDTH - SIDE_HEADER_CELL_WIDTH * 2;

const FOOTER_LEFT_HEIGHT = 3;
const FOOTER_LEFT_WIDTH = 1;
const FOOTER_LEFT_OFFSET_X = 2;
const FOOTER_LEFT_OFFSET_Y = 2;

const FOOTER_RIGHT_HEIGHT = 4;
const FOOTER_RIGHT_WIDTH = 7;
const FOOTER_RIGHT_OFFSET_X = 2;
const FOOTER_RIGHT_OFFSET_Y = 2;

export default function Home() {
  const { gridDim } = useGlobalContext() as { gridDim: Dim2D };

  return (
    <>
      {/* Header */}
      <div
        style={{
          gridRowStart: HEADER_OFFSET_Y,
          gridRowEnd: HEADER_OFFSET_Y + HEADER_CELL_HEIGHT,
          gridColumnStart: gridDim.x / 2 - HEADER_CELL_WIDTH / 2 + 1,
          gridColumnEnd: gridDim.x / 2 + HEADER_CELL_WIDTH / 2 + 1,
        }}
      >
        {/* HEADER CELL */}
        <div
          className="w-full h-full grid"
          style={{
            gridTemplateColumns: `repeat(${HEADER_CELL_WIDTH}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${HEADER_CELL_HEIGHT}, minmax(0, 1fr))`,
          }}
        >
          <div
            className={"bg-red-500 row-span-full"}
            style={{
              gridColumnStart: 1,
              gridColumnEnd: 1 + SIDE_HEADER_CELL_WIDTH,
              gridRowStart: 1,
              gridRowEnd: HEADER_CELL_HEIGHT - 1,
            }}
          >
            {/* <button className="w-full bg-black">about</button> */}
          </div>
          <div
            className={`bg-green-500 row-span-full`}
            style={{
              gridColumnStart: 1 + SIDE_HEADER_CELL_WIDTH,
              gridColumnEnd:
                1 + SIDE_HEADER_CELL_WIDTH + CENTER_HEADER_CELL_WIDTH,
              gridRowStart: 1,
              gridRowEnd: HEADER_CELL_HEIGHT - 1,
            }}
          >
            {/* <div className="w-full h-full">Put logo button here</div> */}
          </div>
          <div
            className={`bg-blue-500 row-span-full`}
            style={{
              gridColumnStart: HEADER_CELL_WIDTH - SIDE_HEADER_CELL_WIDTH + 1,
              gridColumnEnd: HEADER_CELL_WIDTH + 1,
              gridRowStart: 1,
              gridRowEnd: HEADER_CELL_HEIGHT - 1,
            }}
          >
            {/* <button className="w-full bg-black">index</button> */}
          </div>
          <div
            className={`bg-yellow-500 row-span-full`}
            style={{
              gridColumnStart: 1,
              gridColumnEnd: HEADER_CELL_WIDTH + 1,
              gridRowStart: HEADER_CELL_HEIGHT - 1,
              gridRowEnd: HEADER_CELL_HEIGHT + 1,
            }}
          >
            {/* <button className="w-full bg-black">index</button> */}
          </div>
        </div>
      </div>
      {/* Footer Left */}
      <div
        className="grid bg-red-500"
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
      ></div>
      {/* Footer Right */}
      <div
        className="grid bg-blue-500"
        style={{
          gridColumnStart:
            gridDim.x - (FOOTER_RIGHT_WIDTH + FOOTER_RIGHT_OFFSET_X),
          gridColumnEnd:
            gridDim.x -
            (FOOTER_RIGHT_WIDTH + FOOTER_RIGHT_OFFSET_X - 1) +
            FOOTER_RIGHT_WIDTH,
          gridRowStart:
            gridDim.y - (FOOTER_RIGHT_HEIGHT + (FOOTER_RIGHT_OFFSET_Y - 1)),
          gridRowEnd:
            gridDim.y -
            (FOOTER_RIGHT_HEIGHT + (FOOTER_RIGHT_OFFSET_Y - 1)) +
            FOOTER_RIGHT_HEIGHT,
          gridTemplateColumns: `repeat(${FOOTER_RIGHT_WIDTH}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${FOOTER_RIGHT_HEIGHT}, minmax(0, 1fr))`,
        }}
      ></div>
    </>
  );
}
