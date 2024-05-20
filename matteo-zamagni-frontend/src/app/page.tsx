"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild, getAbsGridCoords } from "@/components/GridChild";
import { HEADER_OFFSET_Y, TOTAL_HEADER_HEIGHT } from "@/components/Header";
import {
  Plus,
  SelectableIconComponent,
  TriangleDown,
} from "@/components/Icons";
import { findNearestCornerOfRect, tronPath } from "@/helpers/gridHelpers";
import { useGridLineAnimation } from "@/hooks/useGridLineAnimation";
import { useGridRectAnimation } from "@/hooks/useGridRectAnimation";
import { useOnNavigate } from "@/hooks/useOnNavigate";
import { TARGET_CELL_SIZE } from "@/hooks/useScreenDim";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Pos2D } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { TypeAnimation } from "react-type-animation";

const CONTENT_GRID_PADDING_X = 6;
const CONTENT_GRID_PADDING_Y = 2;

enum HomepageItemType {
  EXHIBITION = "exhibition",
  PROJECT = "project",
}

const homepageItemArray = [
  HomepageItemType.EXHIBITION,
  HomepageItemType.PROJECT,
];

const HomepageItemTypeIconMap: Record<
  HomepageItemType,
  SelectableIconComponent
> = {
  [HomepageItemType.EXHIBITION]: Plus,
  [HomepageItemType.PROJECT]: TriangleDown,
};

type HomepageItem = {
  position: { x: number; y: number };
  type: HomepageItemType;
  title: string;
  year: string;
  imageSrc: string;
  slug: string;
  tags?: string[];
};

const DUMMY_HOMEPAGE_ITEMS: HomepageItem[] = [
  {
    position: { x: 0.2, y: 0.2 },
    type: HomepageItemType.EXHIBITION,
    title: "HELLO TESTING",
    year: "2022",
    imageSrc: "/placeholder.png",
    slug: "testing-1",
    tags: ["ar", "vr", "moving image"],
  },
  {
    position: { x: 0.4, y: 0.1 },
    type: HomepageItemType.PROJECT,
    title: "HELLO TESTING 2",
    year: "2021",
    imageSrc: "/placeholder.png",
    slug: "testing-2",
    tags: ["tag 1", "tag 5"],
  },
  {
    position: { x: 0.9, y: 0.8 },
    type: HomepageItemType.PROJECT,
    title: "HELLO TESTING 3",
    year: "2021",
    imageSrc: "/placeholder.png",
    slug: "testing-3",
    tags: ["tag 1", "tag 5"],
  },
  {
    position: { x: 0.1, y: 0.6 },
    type: HomepageItemType.PROJECT,
    title: "HELLO TESTING 4",
    year: "2021",
    imageSrc: "/placeholder.png",
    slug: "testing-4",
  },
  {
    position: { x: 0.7, y: 0.2 },
    type: HomepageItemType.EXHIBITION,
    title: "HELLO TESTING 5",
    year: "2015",
    imageSrc: "/placeholder.png",
    slug: "testing-4",
  },
];

const QUADRANT_PADDING = { x: 2, y: 2 };

export default function Home() {
  const dispatch = useGlobalContextDispatch();
  const { gridDim, grid, selectedYear } = useGlobalContext();

  const [selectedItemTitle, setSelectedItemTitle] = useState<string | null>(
    null
  );

  const { startRectAnimation, clearRect: clearRectAnimation } =
    useGridRectAnimation();
  const { startAnimation, cancelAnimation: cancelDiagonalAnimation } =
    useGridLineAnimation();

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

  const getImagePos = useCallback(
    (itemPos: HomepageItem["position"]) => {
      // TODO: Make this dynamic
      if (centerContainerVals) {
        const width = centerContainerVals.width / 2 - QUADRANT_PADDING.x * 2;
        const height = centerContainerVals.height / 2 - QUADRANT_PADDING.y * 2;
        return {
          x:
            itemPos.x < 0.5
              ? centerContainerVals.width / 2
              : centerContainerVals.width / 2 - QUADRANT_PADDING.x - width,
          y:
            itemPos.y < 0.5
              ? centerContainerVals.height / 2
              : centerContainerVals.height / 2 - QUADRANT_PADDING.y - height,
          width,
          height,
        };
      }
    },
    [centerContainerVals]
  );

  const selectedItem = useMemo(() => {
    return (
      DUMMY_HOMEPAGE_ITEMS.find((item) => item.title === selectedItemTitle) ||
      null
    );
  }, [selectedItemTitle]);

  const selectedItemImagePos = useMemo(() => {
    if (selectedItem) {
      const imageGridPos = getImagePos(selectedItem.position);
      return imageGridPos;
    }
    return null;
  }, [selectedItem, getImagePos]);

  const selectedItemDescriptionPos = useMemo(() => {
    if (selectedItem && centerContainerVals && gridDim) {
      // TODO: Make this dynamic
      const width = 9;
      const height = 3;
      const absPos = getAbsGridCoords(
        { x: centerContainerVals.width, y: centerContainerVals.height },
        selectedItem.position
      );
      return {
        x: selectedItem.position.x < 0.5 ? absPos.x + 1 : absPos.x - width - 2,
        y:
          selectedItem.position.y < 0.5
            ? absPos.y - height < 0
              ? absPos.y - height / 2 < 0
                ? absPos.y
                : absPos.y - height / 2
              : absPos.y - height
            : absPos.y,
        width,
        height,
      };
    }
    return null;
  }, [centerContainerVals, selectedItem, gridDim]);

  const handleIconClick = useCallback(
    (item: HomepageItem) => {
      if (dispatch && grid) {
        clearRectAnimation();
        cancelDiagonalAnimation();
        if (selectedItemTitle === item.title) {
          setSelectedItemTitle(null);
        } else {
          setSelectedItemTitle(item.title);
          const imagePos = getImagePos(item.position);
          if (imagePos && centerContainerVals) {
            const ledRect = {
              x: centerContainerVals.x + imagePos.x - 1,
              y: centerContainerVals.y + imagePos.y - 1,
              width: imagePos.width + 1,
              height: imagePos.height + 1,
            };
            startRectAnimation(
              ledRect.x,
              ledRect.y,
              ledRect.width,
              ledRect.height
            );
            // Here, point coords are proportional (0-1).
            // We need to scale to absolute grid coordinates to be able to run
            // the tronPath algo.
            if (gridDim) {
              // These are for the whole grid.
              const absPointCoords = getAbsGridCoords(
                { x: centerContainerVals.width, y: centerContainerVals.height },
                item.position
              );
              // But our points are positioned within the center grid. So need to offset
              const outerGridPointCoords: Pos2D = {
                x: absPointCoords.x + centerContainerVals.x - 1,
                y: absPointCoords.y + centerContainerVals.y - 1,
              };
              const outerGridImagePos = {
                ...imagePos,
                x: imagePos.x + centerContainerVals.x,
                y: imagePos.y + centerContainerVals.y,
              };
              const nearestImageCorner = findNearestCornerOfRect(
                outerGridPointCoords, // point of icon click
                outerGridImagePos // rect info
              );
              const path = tronPath(outerGridPointCoords, nearestImageCorner);
              startAnimation(path, 1000);
            }
          }
        }
      }
    },
    [
      dispatch,
      grid,
      selectedItemTitle,
      startRectAnimation,
      clearRectAnimation,
      getImagePos,
      centerContainerVals,
      gridDim,
      startAnimation,
      cancelDiagonalAnimation,
    ]
  );

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
      dispatch({ type: "SET_IS_DARK", val: true });
    }
  }, [dispatch]);

  useEffect(() => {
    if (
      selectedYear &&
      selectedYear !== selectedItem?.year &&
      cancelDiagonalAnimation &&
      clearRectAnimation
    ) {
      clearRectAnimation();
      cancelDiagonalAnimation();
      setSelectedItemTitle(null);
    }
  }, [selectedItem, selectedYear, clearRectAnimation, cancelDiagonalAnimation]);

  const handleNavigate = useCallback(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
    }
    cancelDiagonalAnimation();
    clearRectAnimation();
  }, [dispatch, cancelDiagonalAnimation, clearRectAnimation]);

  useOnNavigate(handleNavigate);

  return (
    <>
      {centerContainerVals && (
        <GridChild {...centerContainerVals} className="">
          {DUMMY_HOMEPAGE_ITEMS.map((item) => {
            const Icon = HomepageItemTypeIconMap[item.type];
            return (
              <GridChild
                outerGridSize={{
                  height: centerContainerVals.height,
                  width: centerContainerVals.width,
                }}
                key={item.title}
                height={1}
                width={1}
                posType="prop"
                {...item.position}
              >
                <button
                  onClick={() => {
                    handleIconClick(item);
                  }}
                  className={`relative w-full h-full flex items-center justify-center overflow-visible ${
                    selectedYear === null || item.year === selectedYear
                      ? ""
                      : "pointer-events-none"
                  }`}
                >
                  <Icon
                    strokeWidth={4}
                    selected={selectedItemTitle === item.title}
                    className={`icon-hover-glow hover:scale-125 transition-all duration-500
                      ${
                        selectedItemTitle
                          ? item.title === selectedItemTitle
                            ? "stroke-highlight"
                            : "stroke-landingIconInactive"
                          : "stroke-white"
                      } ${
                      selectedYear === null || item.year === selectedYear
                        ? selectedYear !== null
                          ? "transition-all scale-110"
                          : ""
                        : "!stroke-landingIconInactive"
                    }
                    `}
                  />
                </button>
              </GridChild>
            );
          })}
          {/* Selected Item Image */}
          <AnimatePresence mode="sync">
            {selectedItem && selectedItemImagePos && (
              <>
                <GridChild
                  {...selectedItemImagePos}
                  isGrid={false}
                  className="relative"
                >
                  <motion.div
                    className={`w-full h-full`}
                    initial={{ opacity: 0 }}
                    exit={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ type: "ease-in-out", duration: 0.5 }}
                    key={selectedItemTitle}
                  >
                    <Image
                      src={selectedItem.imageSrc}
                      className="object-cover w-full h-full"
                      alt=""
                      width={selectedItemImagePos.width * TARGET_CELL_SIZE}
                      height={selectedItemImagePos.height * TARGET_CELL_SIZE}
                    />
                  </motion.div>
                </GridChild>
              </>
            )}
          </AnimatePresence>
          <AnimatePresence mode="sync">
            {selectedItem &&
              selectedItemDescriptionPos &&
              selectedItemImagePos && (
                <>
                  <GridChild
                    {...selectedItemDescriptionPos}
                    isGrid={false}
                    className="relative"
                  >
                    <motion.div
                      className={`w-full h-full text-left bg-black`}
                      initial={{ opacity: 0 }}
                      exit={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ type: "ease-in-out", duration: 0.5 }}
                      key={selectedItemTitle}
                    >
                      <Link
                        href={`/project/${selectedItem.slug}`}
                        className="flex h-full w-full"
                      >
                        <div className="w-[90%] h-full flex flex-col justify-between">
                          <TypeAnimation
                            sequence={[selectedItem.title]}
                            wrapper="span"
                            speed={50}
                            style={{ display: "inline-block" }}
                          />
                          <div className="text-xs">
                            <p className="width-fit">
                              {selectedItem.tags
                                ? selectedItem.tags.join(", ")
                                : null}
                            </p>
                          </div>
                        </div>
                        <div className="w-[10%] h-full flex">
                          <div className="flex text-white items-center justify-center w-full h-full animate-arrowGesture animation-delay-500">
                            <p className="text-xs">{">"}</p>
                          </div>
                        </div>
                      </Link>
                    </motion.div>
                  </GridChild>
                </>
              )}
          </AnimatePresence>
        </GridChild>
      )}
      <FooterRight footerRightHeight={4} footerRightWidth={6}>
        <GridChild
          x={0}
          y={0}
          width={6}
          height={4}
          className="w-full h-full text-[12px]"
        >
          <GridChild
            x={0}
            y={0}
            width={6}
            height={1}
            innerGridWidth={1}
            className="border-b-[1px] border-white"
          >
            <p className="translate-y-[-4px]">legend</p>
          </GridChild>
          <GridChild
            x={0}
            y={1}
            width={6}
            height={4 - 1}
            innerGridHeight={1}
            innerGridWidth={1}
          >
            <div className="w-full h-full flex flex-col justify-around items-start">
              {homepageItemArray.map((type) => {
                const Icon = HomepageItemTypeIconMap[type];
                return (
                  <div key={type} className="w-full h-4 flex">
                    <Icon
                      className="w-4 mr-4 translate-y-[2px]"
                      stroke="white"
                      selected={false}
                    />
                    <p>{type}</p>
                  </div>
                );
              })}
            </div>
          </GridChild>
        </GridChild>
      </FooterRight>
    </>
  );
}
