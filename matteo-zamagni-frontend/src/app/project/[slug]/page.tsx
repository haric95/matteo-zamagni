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

type ProjectData = {
  text: { text: string };
  images: {
    thumbnailURL: string;
    imageURL: string;
    alt: string;
  }[];
  video?: { url: string };
};

enum ProjectMode {
  TEXT,
  IMAGES,
  VIDEO,
}

const DUMMY_DATA: ProjectData = {
  text: {
    text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin hendrerit lorem et felis condimentum elementum non a nunc. Duis maximus nunc sit amet nisl cursus, sit amet efficitur purus rutrum. Pellentesque egestas egestas velit nec posuere. In faucibus dui ut placerat viverra. Nam sollicitudin aliquam orci id egestas. Morbi tempus euismod porttitor. Donec fringilla euismod lectus, condimentum laoreet mauris fermentum vel. Donec molestie gravida scelerisque. Aenean scelerisque egestas mauris sed sagittis. Praesent metus eros, cursus ac eleifend eu, lobortis at dolor. Mauris magna lacus, egestas eget est vitae, blandit laoreet quam. Nam quis faucibus eros. Morbi consequat est in libero congue consequat. Cras placerat nibh eget ligula luctus dignissim. Ut pretium nisi nunc, eget condimentum libero cursus ac. Morbi sed purus imperdiet, iaculis lacus sit amet, accumsan quam. In lorem metus, finibus in sodales eu, aliquam ut nibh. Vivamus sagittis, mi sed tristique vehicula, metus tellus viverra arcu, ut ultrices orci ante non elit. Maecenas interdum, ante non posuere consequat, metus nisl varius urna, id mattis dolor tortor eu augue. Cras nec efficitur dui. Quisque eu ex odio. Donec pretium bibendum mi porttitor ultricies. Pellentesque vehicula sapien in ex scelerisque, at vehicula libero venenatis. Proin ullamcorper ullamcorper ligula, nec vestibulum purus blandit facilisis. In mattis rutrum justo et posuere. Nulla elementum imperdiet mi, et condimentum libero cursus eget. In ac justo ac metus consequat viverra. Cras rutrum leo at venenatis scelerisque.",
  },
  images: [
    {
      thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
      imageURL: "https://placehold.co/640x480/EEE/ff0000",
      alt: "image description",
    },
    {
      thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
      imageURL: "https://placehold.co/640x480/EEE/ff0000",
      alt: "image description",
    },
    {
      thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
      imageURL: "https://placehold.co/640x480/EEE/ff0000",
      alt: "image description",
    },
    {
      thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
      imageURL: "https://placehold.co/640x480/EEE/ff0000",
      alt: "image description",
    },
    {
      thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
      imageURL: "https://placehold.co/640x480/EEE/ff0000",
      alt: "image description",
    },
  ],
};

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

  useEffect(() => {}, []);

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
        <div className="w-full h-full overflow-auto">
          {DUMMY_DATA.text.text}
        </div>
      </div>
      <FooterRight footerRightHeight={5} footerRightWidth={6}>
        <div
          className="grid col-span-full row-span-full translate-y-[8px]"
          style={{
            gridTemplateColumns: `repeat(${6}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${8}, minmax(0, 1fr))`,
          }}
        >
          <div className="col-span-full row-span-1 flex items-start border-white border-b-[1px]">
            <p className="text-[12px] translate-y-[-12px]">navigation</p>
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
                      ? "text-white"
                      : "text-textInactive"
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
                      ? "text-white"
                      : "text-textInactive"
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
                      ? "text-white"
                      : "text-textInactive"
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
