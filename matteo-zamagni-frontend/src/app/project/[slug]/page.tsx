"use client";
import { FooterRight } from "@/components/FooterRight";
import { drawVerticalLine } from "@/helpers/gridHelpers";
import { useOnNavigate } from "@/hooks/useOnNavigate";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Grid } from "@/types/global";
import { useCallback, useEffect, useMemo, useState } from "react";

const CENTER_CELL_WIDTH_PROPOPRTION = 0.4;
const CENTER_CELL_HEIGHT_PROPORTION = 0.5;
const CENTER_CELL_OFFSET_PROPORTION = 0.05;

enum ProjectMode {
  TEXT,
  IMAGES,
  VIDEO,
}

// TODO: Add on mount delay to wait until bg color change has happened
// TODO: Add About Modes
export default function Home() {
  const { gridDim, grid } = useGlobalContext() as {
    gridDim: Dim2D;
    grid: Grid;
  };
  const dispatch = useGlobalContextDispatch();

  const [projectMode, setProjectMode] = useState<ProjectMode>(ProjectMode.TEXT);
  const [ledIsSet, setLedIsSet] = useState(false);

  const centerCellPos = useMemo(() => {
    const width =
      Math.floor(gridDim.x * 0.5 * CENTER_CELL_WIDTH_PROPOPRTION) * 2;
    const height =
      Math.floor(gridDim.y * 0.5 * CENTER_CELL_HEIGHT_PROPORTION) * 2;

    const yCenterOffest = Math.floor(gridDim.y * CENTER_CELL_OFFSET_PROPORTION);

    return {
      colStart: 1 + gridDim.x / 2 - width / 2,
      colEnd: 1 + gridDim.x / 2 + width / 2,
      rowStart: 1 + gridDim.y / 2 + yCenterOffest - height / 2,
      rowEnd: 1 + gridDim.y / 2 + yCenterOffest + height / 2,
    };
  }, [gridDim]);

  useEffect(() => {
    if (gridDim) {
      setLedIsSet(false);
    }
  }, [gridDim]);

  useEffect(() => {
    const updatedGrid = drawVerticalLine(
      drawVerticalLine(
        grid,
        // grid is 0 indexed and we want to highlight the column on the outside of the box
        centerCellPos.colStart - 2,
        centerCellPos.rowStart - 1,
        centerCellPos.rowEnd - 1
      ),
      centerCellPos.colEnd - 1,
      centerCellPos.rowStart - 1,
      centerCellPos.rowEnd - 1
    );
    if (dispatch && !ledIsSet) {
      setLedIsSet(true);
      dispatch({ type: "CLEAR_GRID" });
      dispatch({ type: "UPDATE_GRID", grid: updatedGrid });
    }
  }, [centerCellPos, grid, dispatch, ledIsSet]);

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "SET_IS_DARK", val: true });
    }
  }, [dispatch]);

  return (
    <>
      <div
        className=""
        style={{
          gridColumnStart: centerCellPos.colStart,
          gridColumnEnd: centerCellPos.colEnd,
          gridRowStart: centerCellPos.rowStart,
          gridRowEnd: centerCellPos.rowEnd,
          // gridTemplateColumns: `repeat(${SIDE_HEADER_CELL_WIDTH}, minmax(0, 1fr))`,
          // gridTemplateRows: `repeat(${HEADER_UPPER_HEIGHT}, minmax(0, 1fr))`,
        }}
      >
        <div className="w-full h-full overflow-auto"></div>
      </div>
      <FooterRight footerRightHeight={8} footerRightWidth={6}>
        <div
          className="grid col-span-full row-span-full  "
          style={{
            gridTemplateColumns: `repeat(${6}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${8}, minmax(0, 1fr))`,
          }}
        >
          <div className="col-span-full row-span-1 flex items-start border-white border-b-[1px]">
            <p className="text-[12px]">navigation</p>
          </div>
          <div
            className={`col-span-full flex items-start`}
            style={{
              gridRowStart: 2,
              gridRowEnd: 100,
            }}
          >
            <div className="w-full h-full flex flex-col justify-center items-start">
              <div className="w-1/2 h-full flex flex-col justify-around items-start py-2">
                <button
                  className={`text-[12px] block ${
                    projectMode === ProjectMode.TEXT
                      ? "text-black"
                      : "text-white"
                  }`}
                  onClick={() => {
                    setProjectMode(ProjectMode.TEXT);
                  }}
                >
                  text
                </button>
                <button
                  className={`text-[12px] block ${
                    projectMode === ProjectMode.IMAGES
                      ? "text-black"
                      : "text-white"
                  }`}
                  onClick={() => {
                    setProjectMode(ProjectMode.IMAGES);
                  }}
                >
                  images
                </button>
                <button
                  className={`text-[12px] block ${
                    projectMode === ProjectMode.VIDEO
                      ? "text-black"
                      : "text-white"
                  }`}
                  onClick={() => {
                    setProjectMode(ProjectMode.VIDEO);
                  }}
                >
                  video
                </button>
              </div>
            </div>
          </div>
        </div>
      </FooterRight>
    </>
  );
}
