"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild } from "@/components/GridChild";
import { HEADER_OFFSET_Y, TOTAL_HEADER_HEIGHT } from "@/components/Header";
import {
  BackChevrons,
  Circle,
  HorizontalLines,
  Plus,
  Star,
  TriangleDown,
} from "@/components/Icons";
import { useOnNavigate } from "@/hooks/useOnNavigate";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Grid } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";
import { MdClose } from "react-icons/md";
import { DEFAULT_ANIMATE_MODE } from "@/const";
import { useTheme } from "@/hooks/useTheme";

enum WorkIndexType {
  PROJECT = "project",
  EXHIBITION = "exhibition",
  INSTALLATION = "installation",
  PERFORMANCE = "performance",
  FILM = "film",
  PRINT = "print",
}

const WORK_INDEX_TYPE_ARRAY = [
  WorkIndexType.PROJECT,
  WorkIndexType.EXHIBITION,
  WorkIndexType.INSTALLATION,
  WorkIndexType.PERFORMANCE,
  WorkIndexType.FILM,
  WorkIndexType.PRINT,
];

const WorkIndexTypeIcon = {
  [WorkIndexType.PROJECT]: Plus,
  [WorkIndexType.EXHIBITION]: TriangleDown,
  [WorkIndexType.INSTALLATION]: HorizontalLines,
  [WorkIndexType.PERFORMANCE]: Circle,
  [WorkIndexType.FILM]: Star,
  [WorkIndexType.PRINT]: BackChevrons,
};

type IndexItem = {
  title: string;
  tags: WorkIndexType[];
};

const DUMMY_INDEX_ITEMS: IndexItem[] = new Array(10)
  .fill([
    {
      title: "Ambient Occlusion",
      tags: [WorkIndexType.EXHIBITION, WorkIndexType.INSTALLATION],
    },
    {
      title: "Anise Gallery",
      tags: [WorkIndexType.INSTALLATION, WorkIndexType.PERFORMANCE],
    },
    {
      title: "Arebyte Gallery",
      tags: [
        WorkIndexType.PRINT,
        WorkIndexType.PROJECT,
        WorkIndexType.FILM,
        WorkIndexType.INSTALLATION,
      ],
    },
  ])
  .flat();

const CONTENT_GRID_PADDING_X = 8;
const CONTENT_GRID_PADDING_Y = 2;

// TODO: Add on mount delay to wait until bg color change has happened
// TODO: Find a way to prevent content being hidden on smaller screen sizes
export default function Index() {
  const { gridDim } = useGlobalContext() as {
    gridDim: Dim2D;
    grid: Grid;
  };
  const dispatch = useGlobalContextDispatch();
  const [selectedType, setSelectedType] = useState<WorkIndexType | null>(null);

  const handleFilterClick = (type: WorkIndexType) => {
    if (selectedType === type) {
      setSelectedType(null);
    } else {
      setSelectedType(type);
    }
  };

  const centerContainerVals = useMemo(() => {
    if (gridDim) {
      return {
        x: CONTENT_GRID_PADDING_X,
        y: HEADER_OFFSET_Y + TOTAL_HEADER_HEIGHT + CONTENT_GRID_PADDING_Y,
        width: gridDim.x - CONTENT_GRID_PADDING_X * 2,
        height:
          gridDim.y -
          HEADER_OFFSET_Y -
          TOTAL_HEADER_HEIGHT -
          CONTENT_GRID_PADDING_Y * 2,
      };
    }
    return null;
  }, [gridDim]);

  const splitIndexItems = useMemo(() => {
    if (centerContainerVals) {
      const leftIndexItems = DUMMY_INDEX_ITEMS.slice(
        0,
        centerContainerVals.height
      );
      const rightIndexItems = DUMMY_INDEX_ITEMS.slice(
        centerContainerVals.height,
        centerContainerVals.height * 2
      );
      return { leftIndexItems, rightIndexItems };
    }
    return null;
  }, [centerContainerVals]);

  const { shouldMount } = useTheme({ isDark: false });

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
    }
  }, [dispatch]);

  const clearGrid = useCallback(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
    }
  }, [dispatch]);

  useOnNavigate(clearGrid);

  return (
    <>
      {centerContainerVals && splitIndexItems && (
        <GridChild {...centerContainerVals}>
          <GridChild
            x={0}
            y={0}
            width={centerContainerVals.width / 2}
            innerGridWidth={1}
            height={centerContainerVals.height}
          >
            {splitIndexItems.leftIndexItems.map((item, index) => {
              return (
                <GridChild
                  key={item.title}
                  x={0}
                  y={index}
                  height={1}
                  width={1}
                  className="flex items-center justify-center px-16"
                  isGrid={false}
                >
                  <div className="flex w-24 h-full justify-start items-center bg-background_Light">
                    {item.tags.map((tag, index) => {
                      const Icon = WorkIndexTypeIcon[tag];
                      return (
                        index < 3 && (
                          <Icon
                            key={tag}
                            strokeWidth={8}
                            className={`w-4 h-4 mr-1 transition-all duration-500 ${
                              selectedType === tag
                                ? "stroke-highlight"
                                : "stroke-white"
                            }`}
                          />
                        )
                      );
                    })}
                  </div>
                  <div className="w-full">
                    <p
                      className={`transition-all duration-500 text-elipsis overflow-hidden w-fit bg-background_Light ${
                        selectedType &&
                        item.tags.slice(0, 3).includes(selectedType)
                          ? "text-highlight"
                          : "text-white"
                      }`}
                    >
                      {item.title.toUpperCase()}
                    </p>
                  </div>
                </GridChild>
              );
            })}
          </GridChild>
          <GridChild
            x={centerContainerVals.width / 2}
            y={0}
            width={centerContainerVals.width / 2}
            innerGridWidth={1}
            height={centerContainerVals.height}
          >
            {splitIndexItems.rightIndexItems.map((item, index) => {
              return (
                <GridChild
                  key={item.title}
                  x={0}
                  y={index}
                  height={1}
                  width={1}
                  className="flex items-center justify-center px-16"
                  isGrid={false}
                >
                  <div className="flex w-24 h-full justify-start items-center bg-background_Light">
                    {item.tags.map((tag, index) => {
                      const Icon = WorkIndexTypeIcon[tag];
                      return (
                        index < 3 && (
                          <Icon
                            key={tag}
                            strokeWidth={8}
                            className={`w-4 h-4 mr-1 transition-all duration-500 ${
                              selectedType === tag
                                ? "stroke-highlight"
                                : "stroke-white"
                            }`}
                          />
                        )
                      );
                    })}
                  </div>
                  <div className="w-full">
                    <p
                      className={`transition-all duration-500 text-elipsis overflow-hidden w-fit bg-background_Light ${
                        selectedType &&
                        item.tags.slice(0, 3).includes(selectedType)
                          ? "text-highlight"
                          : "text-white"
                      }`}
                    >
                      {item.title.toUpperCase()}
                    </p>
                  </div>
                </GridChild>
              );
            })}
          </GridChild>
        </GridChild>
      )}
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
          <div className="col-span-full row-span-1 flex justify-between items-start border-black border-b-[1px]">
            <p className="text-[12px]">legend</p>
            <AnimatePresence>
              {selectedType !== null && (
                <motion.button
                  {...DEFAULT_ANIMATE_MODE}
                  className="icon-hover-glow transition-all duration-500"
                  onClick={() => {
                    setSelectedType(null);
                  }}
                >
                  <MdClose />
                </motion.button>
              )}
            </AnimatePresence>
          </div>
          <div
            className={`col-span-full flex items-start`}
            style={{
              gridRowStart: 2,
              gridRowEnd: 100,
            }}
          >
            <div className="w-full h-full flex flex-col justify-center items-end">
              <div className="w-full h-full flex flex-col justify-around items-start py-2 text-xs">
                {WORK_INDEX_TYPE_ARRAY.map((indexType) => {
                  const Component =
                    WorkIndexTypeIcon[
                      indexType as keyof typeof WorkIndexTypeIcon
                    ];
                  return (
                    <button
                      key={indexType}
                      onClick={() => handleFilterClick(indexType)}
                      className="w-full h-4 flex items-center transition-all duration-500"
                    >
                      <Component
                        className={`mr-2 w-4 h-4 transition-color duration-500 ${
                          selectedType === indexType
                            ? "stroke-highlight"
                            : "stroke-white"
                        }`}
                        strokeWidth={8}
                      ></Component>
                      <p
                        className={`translate-y-[-1px] transition-color duration-500 ${
                          selectedType === indexType
                            ? "text-highlight"
                            : "white"
                        }`}
                      >
                        {indexType}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </FooterRight>
    </>
  );
}
