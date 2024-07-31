"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild } from "@/components/GridChild";
import { HEADER_OFFSET_Y, TOTAL_HEADER_HEIGHT } from "@/components/Header";
import { MotionGridChild } from "@/components/MotionGridChild";
import {
  DEFAULT_ANIMATE_MODE,
  HomepageItemTypeIconMap,
  homepageItemArray,
} from "@/const";
import { parseTagsString } from "@/helpers/parseTagsString";
import { TABLET_WIDTH, useIsMobile } from "@/hooks/useIsMobile";
import { useOnNavigate } from "@/hooks/useOnNavigate";
import { usePrefetchImages } from "@/hooks/usePrefetchImages";
import { useStrapi } from "@/hooks/useStrapi";
import { useTheme } from "@/hooks/useTheme";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import {
  Dim2D,
  Grid,
  HomepageItemType,
  IndexItem,
  IndexPageData,
  PosAndDim2D,
  WORK_INDEX_TYPE_ARRAY,
  WorkIndexType,
  WorkIndexTypeIcon,
} from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { MdClose } from "react-icons/md";
import { TfiLayoutMenuV } from "react-icons/tfi";

const CONTENT_GRID_PADDING_X = 10;
const CONTENT_GRID_PADDING_Y = 2;
const CONTENT_GRID_PADDING_X_MOBILE = 1;
const CONTENT_GRID_PADDING_Y_MOBILE = 1;

// TODO: Find a way to prevent content being hidden on smaller screen sizes
export default function Index() {
  const indexData = useStrapi<IndexPageData, false>("/homepage", {
    populate: "deep",
  });

  const { gridDim, cellSize, selectedYear } = useGlobalContext() as {
    gridDim: Dim2D;
    grid: Grid;
    cellSize: Dim2D;
    selectedYear: string;
  };

  const [hoveredItemImageUrl, setHoveredItemImageUrl] = useState<string | null>(
    null
  );

  const isTablet = useIsMobile(TABLET_WIDTH);
  const paddingX = isTablet
    ? CONTENT_GRID_PADDING_X_MOBILE
    : CONTENT_GRID_PADDING_X;
  const paddingY = isTablet
    ? CONTENT_GRID_PADDING_Y_MOBILE
    : CONTENT_GRID_PADDING_Y;

  const dispatch = useGlobalContextDispatch();
  const [selectedType, setSelectedType] = useState<
    WorkIndexType | HomepageItemType | null
  >(null);

  const handleFilterClick = (type: typeof selectedType) => {
    if (selectedType === type) {
      setSelectedType(null);
    } else {
      setSelectedType(type);
      if (dispatch) dispatch({ type: "SET_MOBILE_FOOTER_MENU", isOpen: false });
    }
  };

  const centerContainerVals = useMemo(() => {
    if (gridDim) {
      return {
        x: paddingX,
        y:
          HEADER_OFFSET_Y + TOTAL_HEADER_HEIGHT + paddingY + (isTablet ? 1 : 0),
        width: gridDim.x - paddingX * 2,
        height:
          gridDim.y -
          HEADER_OFFSET_Y -
          TOTAL_HEADER_HEIGHT -
          paddingY * 2 -
          (isTablet ? 5 : 0),
      };
    }
    return null;
  }, [gridDim, paddingX, paddingY, isTablet]);

  const splitIndexItems = useMemo(() => {
    if (indexData && centerContainerVals) {
      const leftIndexItems: IndexItem[] = [];
      const rightIndexItems: IndexItem[] = [];
      indexData.data.attributes.items.forEach((item, index) => {
        if (index % 2 === 0) {
          leftIndexItems.push(item);
        } else {
          rightIndexItems.push(item);
        }
      });
      return { leftIndexItems, rightIndexItems };
    }
    return null;
  }, [centerContainerVals, indexData]);

  const thumbnailViewerPosition = useMemo<PosAndDim2D | null>(() => {
    if (gridDim && !isTablet) {
      return { x: 2, y: gridDim.y - 10, width: 4, height: 4 };
    }
    return null;
  }, [gridDim, isTablet]);

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

  usePrefetchImages(
    indexData?.data.attributes.items.map(
      (item) => item.image.thumbnail.data.attributes.url
    ) ?? null
  );

  return (
    <AnimatePresence>
      {centerContainerVals && splitIndexItems && shouldMount && (
        <MotionGridChild {...DEFAULT_ANIMATE_MODE} {...centerContainerVals}>
          <GridChild
            x={0}
            y={0}
            width={centerContainerVals.width}
            height={centerContainerVals.height}
            isGrid={false}
            className="overflow-y-scroll flex no-scrollbar"
          >
            {isTablet ? (
              // Mobile
              <div className="w-full h-full flex flex-col text-sm">
                {indexData &&
                  [...indexData.data.attributes.items].map((item, index) => {
                    return (
                      <div
                        key={item.title}
                        style={{ height: cellSize.y }}
                        className="w-full"
                      >
                        <Link
                          href={`/project/${item.slug}`}
                          className={`w-full h-full flex items-center justify-between hover-glow-light transition-all duration-500 ${
                            selectedYear === null ||
                            Number(selectedYear) === Number(item.year)
                              ? ""
                              : "blur-[2px] opacity-25"
                          }`}
                        >
                          <div className="flex w-24 h-full justify-start items-center">
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
                                          : "stroke-offWhite"
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
                                            : "stroke-offWhite"
                                        }`}
                                      />
                                    )
                                  );
                                }
                              })}
                          </div>
                          <div className="w-fit">
                            <p
                              className={`transition-all duration-500 text-elipsis overflow-hidden w-fit ${
                                selectedType &&
                                [
                                  ...parseTagsString(item.tags),
                                  item.type,
                                ].includes(selectedType)
                                  ? "text-offWhite"
                                  : "text-black"
                              }`}
                            >
                              {item.title.toUpperCase()}
                            </p>
                          </div>
                        </Link>
                      </div>
                    );
                  })}
              </div>
            ) : (
              // Desktop
              <>
                <div className="w-1/2 h-full flex flex-col translate-x-[16%]">
                  {[...splitIndexItems.leftIndexItems].map((item, index) => {
                    return (
                      <div
                        key={item.title}
                        style={{ height: cellSize.y }}
                        className="w-full"
                      >
                        <Link
                          href={`/project/${item.slug}`}
                          className={`w-full h-full flex items-center justify-center hover-glow-light transition-all duration-500 ${
                            selectedYear === null ||
                            Number(selectedYear) === Number(item.year)
                              ? ""
                              : "blur-[2px] opacity-25"
                          }`}
                          onMouseEnter={() => {
                            setHoveredItemImageUrl(
                              item.image.thumbnail.data.attributes.url
                            );
                          }}
                          onMouseLeave={() => {
                            setHoveredItemImageUrl(null);
                          }}
                        >
                          <div className="flex w-24 pr-8 h-full justify-end items-center">
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
                                          : "stroke-offWhite"
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
                                            : "stroke-offWhite"
                                        }`}
                                      />
                                    )
                                  );
                                }
                              })}
                          </div>
                          <div className="flex-1">
                            <p
                              className={`transition-all duration-500 ${
                                selectedType &&
                                [
                                  ...parseTagsString(item.tags),
                                  item.type,
                                ].includes(selectedType)
                                  ? "translate-x-2"
                                  : ""
                              } text-elipsis overflow-hidden ${
                                selectedType &&
                                [
                                  ...parseTagsString(item.tags),
                                  item.type,
                                ].includes(selectedType)
                                  ? "text-offWhite"
                                  : "text-black"
                              }`}
                            >
                              {`${"                   "}`}
                              {item.title.toUpperCase()}
                            </p>
                          </div>
                        </Link>
                      </div>
                    );
                  })}
                </div>
                <div className="w-1/2 h-full flex flex-col flex-wrap translate-x-[16%]">
                  {[...splitIndexItems.rightIndexItems].map((item, index) => {
                    return (
                      <div
                        key={item.title}
                        style={{ height: cellSize.y }}
                        className="w-full"
                      >
                        <Link
                          href={`/project/${item.slug}`}
                          className={`w-full h-full flex items-center justify-center hover-glow-light transition-all duration-500 ${
                            selectedYear === null ||
                            Number(selectedYear) === Number(item.year)
                              ? ""
                              : "blur-[2px] opacity-25"
                          }`}
                          onMouseEnter={() => {
                            setHoveredItemImageUrl(
                              item.image.thumbnail.data.attributes.url
                            );
                          }}
                          onMouseLeave={() => {
                            setHoveredItemImageUrl(null);
                          }}
                        >
                          <div className="flex w-24 pr-8 h-full justify-end items-center">
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
                                          : "stroke-offWhite"
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
                                            : "stroke-offWhite"
                                        }`}
                                      />
                                    )
                                  );
                                }
                              })}
                          </div>
                          <div className="flex-1">
                            <p
                              className={`transition-all duration-500 ${
                                selectedType &&
                                [
                                  ...parseTagsString(item.tags),
                                  item.type,
                                ].includes(selectedType)
                                  ? "translate-x-2"
                                  : ""
                              } text-elipsis overflow-hidden w-full ${
                                selectedType &&
                                [
                                  ...parseTagsString(item.tags),
                                  item.type,
                                ].includes(selectedType)
                                  ? "text-offWhite"
                                  : "text-black"
                              }`}
                            >
                              {item.title.toUpperCase()}
                            </p>
                          </div>
                        </Link>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </GridChild>
        </MotionGridChild>
      )}
      {thumbnailViewerPosition && (
        <GridChild {...thumbnailViewerPosition} className="relative">
          {hoveredItemImageUrl && (
            <Image
              className="fade-in"
              src={hoveredItemImageUrl}
              alt="Project thumbnail"
              layout="fill"
              objectFit="cover"
            />
          )}
        </GridChild>
      )}
      <FooterRight
        footerRightHeight={8}
        footerRightWidth={6}
        isMounted={shouldMount}
        breakpointSize={TABLET_WIDTH}
        mobileTitleComponent={
          selectedType ? (
            <button className="text-s flex justify-end items-center">
              {(() => {
                const Component =
                  HomepageItemTypeIconMap[
                    selectedType as keyof typeof HomepageItemTypeIconMap
                  ] ??
                  WorkIndexTypeIcon[
                    selectedType as keyof typeof WorkIndexTypeIcon
                  ];
                return (
                  <>
                    <Component
                      className={`mr-2 w-4 h-4 transition-color duration-500 stroke-highlight`}
                      strokeWidth={8}
                    ></Component>
                    <p>{selectedType}</p>
                  </>
                );
              })()}
            </button>
          ) : (
            <div className="flex items-center">
              <TfiLayoutMenuV color="white" className="mr-1" />
              <p>legend</p>
            </div>
          )
        }
      >
        <div
          className="grid col-span-full row-span-full text-black"
          style={{
            gridTemplateColumns: `repeat(${6}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${8}, minmax(0, 1fr))`,
          }}
        >
          <div className="col-span-full row-span-1 flex justify-between items-start border-white md:border-black border-b-[1px]">
            <p className="text-[12px] md:text-black text-offWhite">legend</p>
            <AnimatePresence>
              {selectedType !== null && (
                <motion.button
                  {...DEFAULT_ANIMATE_MODE}
                  className="icon-hover-glow transition-all duration-500"
                  onClick={() => {
                    setSelectedType(null);
                    if (dispatch) {
                      dispatch({
                        type: "SET_MOBILE_FOOTER_MENU",
                        isOpen: false,
                      });
                    }
                  }}
                >
                  <MdClose color={isTablet ? "white" : "black"} />
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
                            : "stroke-offWhite"
                        }`}
                        strokeWidth={8}
                      ></Component>
                      <p
                        className={`translate-y-[-1px] transition-color duration-500 ${
                          selectedType === indexType
                            ? "text-offWhite md:text-offWhite translate-x-2"
                            : "text-offWhite md:text-black"
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
                            : "stroke-offWhite"
                        }`}
                        strokeWidth={8}
                      ></Component>
                      <p
                        className={`translate-y-[-1px] transition-color duration-500 ${
                          selectedType === indexType
                            ? "text-offWhite md:text-offWhite translate-x-2"
                            : "text-offWhite md:text-black"
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
