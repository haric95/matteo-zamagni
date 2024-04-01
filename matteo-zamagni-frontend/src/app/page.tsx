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
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Pos2D } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";

const CONTENT_GRID_PADDING_X = 6;
const CONTENT_GRID_PADDING_Y = 0;

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
};

const DUMMY_HOMEPAGE_ITEMS: HomepageItem[] = [
  {
    position: { x: 0.2, y: 0.2 },
    type: HomepageItemType.EXHIBITION,
    title: "HELLO TESTING",
    year: "2022",
    imageSrc: "https://placehold.co/600x400/EEE/31343C",
  },
  {
    position: { x: 0.4, y: 0.1 },
    type: HomepageItemType.PROJECT,
    title: "HELLO TESTING 2",
    year: "2021",
    imageSrc: "https://placehold.co/600x400/EEE/31343C",
  },
  {
    position: { x: 0.9, y: 0.8 },
    type: HomepageItemType.PROJECT,
    title: "HELLO TESTING 3",
    year: "2021",
    imageSrc: "https://placehold.co/600x400/EEE/31343C",
  },
  {
    position: { x: 0.1, y: 0.6 },
    type: HomepageItemType.PROJECT,
    title: "HELLO TESTING 4",
    year: "2021",
    imageSrc: "https://placehold.co/600x400/EEE/31343C",
  },
];

export default function Home() {
  const dispatch = useGlobalContextDispatch();
  const { gridDim, grid } = useGlobalContext();

  const [selectedItemTitle, setSelectedItemTitle] = useState<string | null>(
    null
  );

  const { startRectAnimation, clearRect } = useGridRectAnimation();
  const { startAnimation } = useGridLineAnimation();

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
      const QUADRANT_PADDING = { x: 2, y: 2 };
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
    return DUMMY_HOMEPAGE_ITEMS.find(
      (item) => item.title === selectedItemTitle
    );
  }, [selectedItemTitle]);

  const selectedItemImagePos = useMemo(() => {
    if (selectedItem) {
      const imageGridPos = getImagePos(selectedItem.position);
      return imageGridPos;
    }
  }, [selectedItem, getImagePos]);

  const handleIconClick = useCallback(
    (item: HomepageItem) => {
      if (dispatch && grid) {
        if (selectedItemTitle === item.title) {
          setSelectedItemTitle(null);
          clearRect();
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
              console.log(outerGridImagePos);
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
      clearRect,
      getImagePos,
      centerContainerVals,
      gridDim,
      startAnimation,
    ]
  );

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
      dispatch({ type: "SET_IS_DARK", val: true });
    }
  }, [dispatch]);

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
                  className="w-full h-full flex items-center justify-center"
                >
                  <Icon
                    strokeWidth={4}
                    selected={selectedItemTitle === item.title}
                    className={`transition-all duration-500
                      ${
                        selectedItemTitle
                          ? item.title === selectedItemTitle
                            ? "stroke-highlight"
                            : "stroke-landingIconInactive"
                          : "stroke-white"
                      }
                    `}
                  />
                </button>
              </GridChild>
            );
          })}
          {/* Selected Item Image */}
          <AnimatePresence mode="popLayout">
            {selectedItem && selectedItemImagePos && (
              <GridChild
                {...selectedItemImagePos}
                isGrid={false}
                className="relative"
              >
                <motion.div
                  className={`w-full h-full bg-green-500`}
                  initial={{ opacity: 0 }}
                  exit={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ type: "ease-in-out", duration: 0.5 }}
                  key={selectedItemTitle}
                ></motion.div>
              </GridChild>
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
                    <p>projects</p>
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
