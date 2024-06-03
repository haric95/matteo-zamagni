"use client";
import { FooterRight } from "@/components/FooterRight";
import { DEFAULT_ANIMATE_MODE } from "@/const";
import { drawVerticalLine } from "@/helpers/gridHelpers";
import { useOnNavigate } from "@/hooks/useOnNavigate";
import { useStrapi } from "@/hooks/useStrapi";
import { useTheme } from "@/hooks/useTheme";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Grid } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";
import Markdown from "react-markdown";

const CENTER_CELL_WIDTH_PROPOPRTION = 0.4;
const CENTER_CELL_HEIGHT_PROPORTION = 0.5;
const CENTER_CELL_OFFSET_PROPORTION = 0.05;

enum AboutMode {
  BIO = "Bio",
  AWARDS = "Awards",
  RESIDENCIES = "Residencies",
  PERFORMANCES = "Performances",
  SCREENINGS = "Screenings",
  TALKS = "Talks",
}

type AboutPageData = {
  [AboutMode.BIO]: string;
  [AboutMode.AWARDS]: string;
  [AboutMode.RESIDENCIES]: string;
  [AboutMode.PERFORMANCES]: string;
  [AboutMode.SCREENINGS]: string;
  [AboutMode.TALKS]: string;
};

// TODO: Add on mount delay to wait until bg color change has happened
// TODO: Add About Modes
export default function Home() {
  const aboutPageData = useStrapi<AboutPageData, false>("/about", {
    populate: "deep",
  });
  const { gridDim, grid } = useGlobalContext() as {
    gridDim: Dim2D;
    grid: Grid;
  };
  const dispatch = useGlobalContextDispatch();

  const [aboutMode, setAboutMode] = useState<AboutMode>(AboutMode.BIO);
  const [ledIsSet, setLedIsSet] = useState(false);

  const { shouldMount } = useTheme({ isDark: false });

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
    if (dispatch && !ledIsSet && shouldMount) {
      setLedIsSet(true);
      dispatch({ type: "CLEAR_GRID" });
      dispatch({ type: "UPDATE_GRID", grid: updatedGrid });
    }
  }, [centerCellPos, grid, dispatch, ledIsSet, shouldMount]);

  const handleNavigate = useCallback(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
    }
  }, [dispatch]);

  useOnNavigate(handleNavigate);

  return (
    <>
      <AnimatePresence>
        {shouldMount && (
          <motion.div
            {...DEFAULT_ANIMATE_MODE}
            className="bg-background_Light"
            style={{
              gridColumnStart: centerCellPos.colStart,
              gridColumnEnd: centerCellPos.colEnd,
              gridRowStart: centerCellPos.rowStart,
              gridRowEnd: centerCellPos.rowEnd,
              // gridTemplateColumns: `repeat(${SIDE_HEADER_CELL_WIDTH}, minmax(0, 1fr))`,
              // gridTemplateRows: `repeat(${HEADER_UPPER_HEIGHT}, minmax(0, 1fr))`,
            }}
            key={aboutMode}
          >
            <div className="w-full h-full overflow-auto text-black whitespace-break-spaces no-scrollbar">
              <Markdown>
                {aboutPageData && aboutPageData.data.attributes[aboutMode]}
              </Markdown>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <FooterRight
        footerRightHeight={8}
        footerRightWidth={6}
        isMounted={shouldMount}
      >
        <div
          className="grid col-span-full row-span-full  "
          style={{
            gridTemplateColumns: `repeat(${6}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${8}, minmax(0, 1fr))`,
          }}
        >
          <div className="col-span-full row-span-1 flex items-start border-black border-b-[1px] text-black">
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
                  className={`text-[12px] block transition-color duration-500 ${
                    aboutMode === AboutMode.BIO ? "text-white" : "text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.BIO);
                  }}
                >
                  bio
                </button>
                <button
                  className={`text-[12px] block transition-color duration-500 ${
                    aboutMode === AboutMode.AWARDS ? "text-white" : "text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.AWARDS);
                  }}
                >
                  awards
                </button>
                <button
                  className={`text-[12px] block transition-color duration-500 ${
                    aboutMode === AboutMode.RESIDENCIES
                      ? "text-white"
                      : "text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.RESIDENCIES);
                  }}
                >
                  residencies
                </button>
                <button
                  className={`text-[12px] block transition-color duration-500 ${
                    aboutMode === AboutMode.PERFORMANCES
                      ? "text-white"
                      : "text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.PERFORMANCES);
                  }}
                >
                  performances
                </button>
                <button
                  className={`text-[12px] block transition-color duration-500 ${
                    aboutMode === AboutMode.SCREENINGS
                      ? "text-white"
                      : "text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.SCREENINGS);
                  }}
                >
                  screenings
                </button>
                <button
                  className={`text-[12px] block transition-color duration-500 ${
                    aboutMode === AboutMode.TALKS ? "text-white" : "text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.TALKS);
                  }}
                >
                  talks
                </button>
              </div>
            </div>
          </div>
        </div>
      </FooterRight>
    </>
  );
}
