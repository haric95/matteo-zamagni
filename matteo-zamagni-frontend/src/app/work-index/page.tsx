"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild } from "@/components/GridChild";
import { HEADER_OFFSET_Y, TOTAL_HEADER_HEIGHT } from "@/components/Header";
import {
  BackChevrons,
  Circle,
  HorizontalLines,
  Star,
} from "@/components/Icons";
import { MotionGridChild } from "@/components/MotionGridChild";
import { DEFAULT_ANIMATE_MODE } from "@/const";
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
import { MdClose } from "react-icons/md";
import {
  HomepageItemType,
  HomepageItemTypeIconMap,
  homepageItemArray,
} from "../page";
import Link from "next/link";
import { parseTagsString } from "@/helpers/parseTagsString";

enum WorkIndexType {
  INSTALLATION = "Installation",
  PERFORMANCE = "Performance",
  FILM = "Film",
  PRINT = "Print",
}

const WORK_INDEX_TYPE_ARRAY = [
  WorkIndexType.INSTALLATION,
  WorkIndexType.PERFORMANCE,
  WorkIndexType.FILM,
  WorkIndexType.PRINT,
];

const WorkIndexTypeIcon = {
  [WorkIndexType.INSTALLATION]: HorizontalLines,
  [WorkIndexType.PERFORMANCE]: Circle,
  [WorkIndexType.FILM]: Star,
  [WorkIndexType.PRINT]: BackChevrons,
};

type IndexItem = {
  type: HomepageItemType;
  title: string;
  year: string;
  slug: string;
  tags: string;
};

type IndexPageData = {
  items: IndexItem[];
};

const CONTENT_GRID_PADDING_X = 8;
const CONTENT_GRID_PADDING_Y = 2;

// TODO: Find a way to prevent content being hidden on smaller screen sizes
export default function Index() {
  const indexData = useStrapi<IndexPageData, false>("/homepage", {
    populate: "deep",
  });

  const { gridDim } = useGlobalContext() as {
    gridDim: Dim2D;
    grid: Grid;
  };

  const dispatch = useGlobalContextDispatch();
  const [selectedType, setSelectedType] = useState<
    WorkIndexType | HomepageItemType | null
  >(null);

  const handleFilterClick = (type: typeof selectedType) => {
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
    if (indexData && centerContainerVals) {
      const leftIndexItems = indexData.data.attributes.items.slice(
        0,
        centerContainerVals.height
      );
      const rightIndexItems = indexData.data.attributes.items.slice(
        centerContainerVals.height,
        centerContainerVals.height * 2
      );
      return { leftIndexItems, rightIndexItems };
    }
    return null;
  }, [centerContainerVals, indexData]);

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
    <AnimatePresence>
      {centerContainerVals && splitIndexItems && shouldMount && (
        <MotionGridChild {...DEFAULT_ANIMATE_MODE} {...centerContainerVals}>
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
                  className="w-full h-full"
                  isGrid={false}
                >
                  <Link
                    href={`/project/${item.slug}`}
                    className="w-fit h-full flex items-center justify-center hover-glow-light"
                  >
                    <div className="flex w-24 h-full justify-start items-center bg-background_Light">
                      <>
                        {homepageItemArray.map((type) => {
                          if (item.type === type) {
                            const Icon = HomepageItemTypeIconMap[type];
                            return (
                              <Icon
                                key={`${item.title}-type`}
                                strokeWidth={8}
                                className={`w-4 h-4 mr-1 transition-all duration-500 ${
                                  selectedType === type
                                    ? "stroke-highlight"
                                    : "stroke-white"
                                }`}
                              />
                            );
                          }
                        })}
                      </>
                      {item.tags
                        .split(",")
                        .map((item) => item.trim())
                        .map((tag, index) => {
                          if (
                            WorkIndexTypeIcon[
                              tag as keyof typeof WorkIndexTypeIcon
                            ]
                          ) {
                            const Icon =
                              WorkIndexTypeIcon[
                                tag as keyof typeof WorkIndexTypeIcon
                              ];
                            return (
                              index < 2 && (
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
                          }
                        })}
                    </div>
                    <div className="w-full">
                      <p
                        className={`transition-all duration-500 text-elipsis overflow-hidden w-fit bg-background_Light ${
                          selectedType &&
                          [...parseTagsString(item.tags), item.type].includes(
                            selectedType
                          )
                            ? "text-white"
                            : "text-black"
                        }`}
                      >
                        {item.title.toUpperCase()}
                      </p>
                    </div>
                  </Link>
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
                  className="w-full h-full"
                  isGrid={false}
                >
                  <Link
                    href={`/project/${item.slug}`}
                    className="w-fit h-full flex items-center justify-center hover-glow-light"
                  >
                    <div className="flex w-24 h-full justify-start items-center bg-background_Light">
                      <>
                        {homepageItemArray.map((type) => {
                          if (item.type === type) {
                            const Icon = HomepageItemTypeIconMap[type];
                            return (
                              <Icon
                                key={`${item.title}-type`}
                                strokeWidth={8}
                                className={`w-4 h-4 mr-1 transition-all duration-500 ${
                                  selectedType === type
                                    ? "stroke-highlight"
                                    : "stroke-white"
                                }`}
                              />
                            );
                          }
                        })}
                      </>
                      {item.tags
                        .split(",")
                        .map((item) => item.trim())
                        .map((tag, index) => {
                          if (
                            WorkIndexTypeIcon[
                              tag as keyof typeof WorkIndexTypeIcon
                            ]
                          ) {
                            const Icon =
                              WorkIndexTypeIcon[
                                tag as keyof typeof WorkIndexTypeIcon
                              ];
                            return (
                              index < 2 && (
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
                          }
                        })}
                    </div>
                    <div className="w-full">
                      <p
                        className={`transition-all duration-500 text-elipsis overflow-hidden w-fit bg-background_Light ${
                          selectedType &&
                          [
                            ...parseTagsString(item.tags),
                            ...homepageItemArray,
                          ].includes(selectedType)
                            ? "text-white"
                            : "text-black"
                        }`}
                      >
                        {item.title.toUpperCase()}
                      </p>
                    </div>
                  </Link>
                </GridChild>
              );
            })}
          </GridChild>
        </MotionGridChild>
      )}
      <FooterRight
        footerRightHeight={8}
        footerRightWidth={6}
        isMounted={shouldMount}
      >
        <div
          className="grid col-span-full row-span-full text-black"
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
                {homepageItemArray.map((indexType) => {
                  const Component = HomepageItemTypeIconMap[indexType];
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
                            ? "text-white"
                            : "text-black"
                        }`}
                      >
                        {indexType}
                      </p>
                    </button>
                  );
                })}
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
                            ? "text-white"
                            : "text-black"
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
    </AnimatePresence>
  );
}
