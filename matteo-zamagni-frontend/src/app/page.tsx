"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild, getAbsGridCoords } from "@/components/GridChild";
import { HEADER_OFFSET_Y, TOTAL_HEADER_HEIGHT } from "@/components/Header";
import { MotionGridChild } from "@/components/MotionGridChild";
import {
  DEFAULT_ANIMATE_MODE,
  HomepageItemTypeIconMap,
  homepageItemArray,
} from "@/const";
import { findNearestCornerOfRect, tronPath } from "@/helpers/gridHelpers";
import { useGridLineAnimation } from "@/hooks/useGridLineAnimation";
import { useGridRectAnimation } from "@/hooks/useGridRectAnimation";
import { useIsMobile } from "@/hooks/useIsMobile";
import { useOnNavigate } from "@/hooks/useOnNavigate";
import { usePrefetchImages } from "@/hooks/usePrefetchImages";
import { TARGET_CELL_SIZE } from "@/hooks/useScreenDim";
import { useStrapi } from "@/hooks/useStrapi";
import { useTheme } from "@/hooks/useTheme";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import {
  HomepageData,
  HomepageItem,
  HomepageItemType,
  Pos2D,
} from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { MdClose } from "react-icons/md";
import { TfiLayoutMenuV } from "react-icons/tfi";
import { TypeAnimation } from "react-type-animation";

const CONTENT_GRID_PADDING_X = 6;
const CONTENT_GRID_PADDING_Y = 2;
const CONTENT_GRID_PADDING_X_MOBILE = 2;
const CONTENT_GRID_PADDING_Y_MOBILE = 2;

const QUADRANT_PADDING = { x: 2, y: 2 };

export default function Home() {
  const homepageData = useStrapi<HomepageData, false>("/homepage", {
    "populate[items][populate][0]": "position",
    "populate[items][populate][1]": "image",
  });

  usePrefetchImages(
    homepageData?.data.attributes.items.map(
      (item) => item.image.data.attributes.url
    ) || null
  );

  const dispatch = useGlobalContextDispatch();
  const { gridDim, grid, selectedYear } = useGlobalContext();
  const isMobile = useIsMobile();

  const [selectedItemTitle, setSelectedItemTitle] = useState<string | null>(
    null
  );
  const [selectedFilterType, setSelectedFilterType] =
    useState<HomepageItemType | null>(null);

  const { startRectAnimation, clearRect: clearRectAnimation } =
    useGridRectAnimation();
  const { startAnimation, cancelAnimation: cancelDiagonalAnimation } =
    useGridLineAnimation();

  const { shouldMount } = useTheme({ isDark: true });

  const centerContainerVals = useMemo(() => {
    if (gridDim) {
      return {
        x: isMobile ? CONTENT_GRID_PADDING_X_MOBILE : CONTENT_GRID_PADDING_X,
        y:
          HEADER_OFFSET_Y +
          TOTAL_HEADER_HEIGHT +
          (isMobile ? CONTENT_GRID_PADDING_Y_MOBILE : CONTENT_GRID_PADDING_Y),
        width:
          gridDim.x -
          (isMobile ? CONTENT_GRID_PADDING_X_MOBILE : CONTENT_GRID_PADDING_X) *
            2,
        height:
          gridDim.y -
          HEADER_OFFSET_Y -
          TOTAL_HEADER_HEIGHT -
          (isMobile ? CONTENT_GRID_PADDING_Y_MOBILE : CONTENT_GRID_PADDING_Y) *
            2,
      };
    }
    return null;
  }, [gridDim, isMobile]);

  const getImagePos = useCallback(
    (itemPos: HomepageItem["position"]) => {
      // TODO: Make this dynamic
      if (centerContainerVals) {
        const width = isMobile
          ? centerContainerVals.width
          : centerContainerVals.width / 2 - QUADRANT_PADDING.x * 2;
        const height = isMobile
          ? Math.floor(width * 0.76)
          : centerContainerVals.height / 2 - QUADRANT_PADDING.y * 2;
        return {
          x: isMobile
            ? 0
            : itemPos.x < 0.5
            ? centerContainerVals.width / 2
            : centerContainerVals.width / 2 - QUADRANT_PADDING.x - width,
          y: isMobile
            ? itemPos.y < 0.5
              ? centerContainerVals.height / 2
              : 0
            : itemPos.y < 0.5
            ? centerContainerVals.height / 2
            : centerContainerVals.height / 2 - QUADRANT_PADDING.y - height,
          width,
          height,
        };
      }
    },
    [centerContainerVals, isMobile]
  );

  const selectedItem = useMemo(() => {
    return (
      homepageData?.data.attributes.items.find(
        (item) => item.title === selectedItemTitle
      ) || null
    );
  }, [homepageData, selectedItemTitle]);

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
      const width = isMobile ? centerContainerVals.width / 2 : 9;
      const height = isMobile ? 3 : 3;
      const absPos = getAbsGridCoords(
        { x: centerContainerVals.width, y: centerContainerVals.height },
        selectedItem.position
      );
      return {
        x: selectedItem.position.x < 0.5 ? absPos.x + 1 : absPos.x - width - 2,
        y: isMobile
          ? selectedItem.position.y > 0.5
            ? absPos.y > height
              ? absPos.y - height
              : absPos.y + 1
            : absPos.y < centerContainerVals.height - height
            ? absPos.y + 1
            : absPos.y - height - 1
          : selectedItem.position.y < 0.5
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
  }, [centerContainerVals, selectedItem, gridDim, isMobile]);

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
      {centerContainerVals && shouldMount && (
        <MotionGridChild
          {...DEFAULT_ANIMATE_MODE}
          {...centerContainerVals}
          className=""
        >
          {homepageData?.data.attributes.items.map((item) => {
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
                      (selectedYear === null || item.year === selectedYear) &&
                      (selectedFilterType === null ||
                        selectedFilterType === item.type)
                        ? selectedYear !== null
                          ? "transition-all scale-110"
                          : ""
                        : "!stroke-landingIconInactive blur-[1px]"
                    }
                    `}
                  />
                </button>
              </GridChild>
            );
          })}
          {/* Selected Item Image */}
          <AnimatePresence>
            {selectedItem && selectedItemImagePos && (
              <MotionGridChild
                {...DEFAULT_ANIMATE_MODE}
                {...selectedItemImagePos}
                isGrid={false}
                className="relative"
              >
                <button
                  className={`w-full h-full image-hover-glow hover:scale-[101%] duration-500 transition-all duration-500`}
                >
                  <Link href={`/project/${selectedItem.slug}`}>
                    <Image
                      src={selectedItem.image.data.attributes.url}
                      className="object-cover w-full h-full"
                      alt=""
                      width={selectedItemImagePos.width * TARGET_CELL_SIZE}
                      height={selectedItemImagePos.height * TARGET_CELL_SIZE}
                    />
                  </Link>
                </button>
              </MotionGridChild>
            )}
          </AnimatePresence>
          {selectedItem &&
            selectedItemDescriptionPos &&
            selectedItemImagePos && (
              <>
                <GridChild
                  {...selectedItemDescriptionPos}
                  isGrid={false}
                  className="relative"
                >
                  <div
                    className={`w-full h-full text-left bg-black transition-all duration-500 hover:scale-[102%]`}
                  >
                    <Link
                      href={`/project/${selectedItem.slug}`}
                      className="flex h-full w-full"
                    >
                      <div className="group w-[90%] h-full flex flex-col justify-between icon-hover-glow transition-all duration-500">
                        <TypeAnimation
                          sequence={[selectedItem.title]}
                          wrapper="span"
                          speed={50}
                          style={{ display: "inline-block" }}
                          className=""
                        />
                        <div className="text-xs">
                          <p className="width-fit">{selectedItem.tags}</p>
                        </div>
                      </div>
                      <div className="w-[10%] h-full flex">
                        <div className="flex text-white items-center justify-center w-full h-full animate-arrowGesture animation-delay-500">
                          <p className="text-xs">{">"}</p>
                        </div>
                      </div>
                    </Link>
                  </div>
                </GridChild>
              </>
            )}
        </MotionGridChild>
      )}
      <FooterRight
        footerRightHeight={4}
        footerRightWidth={6}
        isMounted={shouldMount}
        mobileTitleComponent={
          selectedFilterType ? (
            <button className="text-s flex justify-end items-center">
              {(() => {
                const Component =
                  HomepageItemTypeIconMap[
                    selectedFilterType as keyof typeof HomepageItemTypeIconMap
                  ];
                return (
                  <>
                    <Component
                      className={`mr-2 w-4 h-4 transition-color duration-500 stroke-highlight`}
                      strokeWidth={8}
                    ></Component>
                    <p>{selectedFilterType}</p>
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
            isGrid={false}
            className="border-b-[1px] border-white flex justify-between items-center"
          >
            <p className="translate-y-[-4px]">legend</p>
            <AnimatePresence>
              {selectedFilterType !== null && (
                <motion.button
                  {...DEFAULT_ANIMATE_MODE}
                  className="icon-hover-glow transition-all duration-50 translate-y-[-4px]"
                  onClick={() => {
                    setSelectedFilterType(null);
                    if (dispatch) {
                      dispatch({
                        type: "SET_MOBILE_FOOTER_MENU",
                        isOpen: false,
                      });
                    }
                  }}
                >
                  <MdClose color={"white"} />
                </motion.button>
              )}
            </AnimatePresence>
          </GridChild>
          <GridChild
            x={0}
            y={1}
            width={6}
            height={3}
            innerGridHeight={1}
            innerGridWidth={1}
          >
            <div className="w-full h-full flex flex-col justify-around items-start">
              {homepageItemArray.map((type) => {
                const Icon = HomepageItemTypeIconMap[type];
                return (
                  <button
                    key={type}
                    className="w-full h-4 flex icon-hover-glow transition-all duration-500"
                    onClick={() => {
                      if (selectedFilterType === type) {
                        setSelectedFilterType(null);
                      } else {
                        setSelectedFilterType(type);
                        if (dispatch) {
                          dispatch({
                            type: "SET_MOBILE_FOOTER_MENU",
                            isOpen: false,
                          });
                        }
                      }
                    }}
                  >
                    <Icon
                      className={`w-4 mr-2 translate-y-[2px]`}
                      stroke="white"
                      selected={false}
                    />
                    <p
                      className={`transition-all duration-500 ${
                        type === selectedFilterType ? "translate-x-2" : ""
                      }`}
                    >
                      {type}
                    </p>
                  </button>
                );
              })}
            </div>
          </GridChild>
        </GridChild>
      </FooterRight>
    </>
  );
}
