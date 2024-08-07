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
import { getImageAspectRatio } from "@/themes/utils";
import {
  HomepageData,
  HomepageItem,
  HomepageItemType,
  PosAndDim2D,
} from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import { set } from "lodash";
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

const QUADRANT_PADDING = { x: 5, y: 5 };

export default function Home() {
  const homepageData = useStrapi<HomepageData, false>("/homepage", {
    populate: "deep",
  });
  const isMobile = useIsMobile();

  const parsedHomepageData = useMemo(() => {
    if (homepageData) {
      if (isMobile) {
        const homepageDataUpdatedItems =
          homepageData?.data.attributes.items.map((item) => ({
            ...item,
            position: { x: item.position.y, y: item.position.x },
          }));
        const object = set(
          homepageData,
          ["data", "attributes", "items"],
          homepageDataUpdatedItems
        );
        return object;
      }
      return homepageData;
    }
    return null;
  }, [homepageData, isMobile]);

  usePrefetchImages(
    parsedHomepageData?.data.attributes.items.map(
      (item) => item.image?.image?.data?.attributes?.url
    ) || null
  );

  const dispatch = useGlobalContextDispatch();
  const { gridDim, selectedYear, scrollerAvailableYears, cellSize } =
    useGlobalContext();

  const [selectedItemTitle, setSelectedItemTitle] = useState<string | null>(
    null
  );
  const [selectedFilterType, setSelectedFilterType] =
    useState<HomepageItemType | null>(null);
  const [selectedItemImagePos, setSelectedItemImagePos] =
    useState<PosAndDim2D | null>(null);

  const { startRectAnimation, clearRect: clearRectAnimation } =
    useGridRectAnimation();

  useEffect(() => {
    if (parsedHomepageData && !scrollerAvailableYears) {
      const years = new Set(["0000"]);
      parsedHomepageData.data.attributes.items.forEach((item) => {
        years.add(String(item.year));
      });
      const sortedYears = Array.from(years).sort(
        (a, b) => Number(a) - Number(b)
      );
      if (dispatch) {
        dispatch({ type: "SET_SCROLLER_AVAILABLE_YEARS", years: sortedYears });
      }
    }
  }, [scrollerAvailableYears, parsedHomepageData, dispatch]);

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

  const selectedItem = useMemo(() => {
    return (
      parsedHomepageData?.data.attributes.items.find(
        (item) => item.title === selectedItemTitle
      ) || null
    );
  }, [parsedHomepageData, selectedItemTitle]);

  const selectedItemDescriptionPos = useMemo(() => {
    if (selectedItem && centerContainerVals && gridDim) {
      // TODO: Make this dynamic
      const width = isMobile ? centerContainerVals.width / 2 : 12;
      const height = isMobile ? 3 : 3;
      const absPos = getAbsGridCoords(
        { x: centerContainerVals.width, y: centerContainerVals.height },
        selectedItem.position
      );
      return {
        x: selectedItem.position.x < 0.5 ? absPos.x + 1 : absPos.x - width - 2,
        y: isMobile
          ? selectedItem.position.y < 0.25
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

  const getImagePos = useCallback(
    async (itemPos: HomepageItem["position"], aspect: number) => {
      if (centerContainerVals && selectedItem) {
        const width = isMobile
          ? centerContainerVals.width
          : centerContainerVals.width / 2 - QUADRANT_PADDING.x * 2;
        const height = Math.floor(width * aspect);
        if (centerContainerVals) {
          const absItemPos = getAbsGridCoords(
            { x: centerContainerVals.width, y: centerContainerVals.height },
            itemPos
          );
          return {
            x: isMobile
              ? 0
              : itemPos.x < 0.5
              ? absItemPos.x + 2
              : absItemPos.x - width - 2,
            y: isMobile
              ? selectedItemDescriptionPos
                ? itemPos.y < 0.25
                  ? selectedItemDescriptionPos.y +
                    selectedItemDescriptionPos.height +
                    2
                  : itemPos.y < 0.75
                  ? absItemPos.y + 2
                  : selectedItemDescriptionPos.y - height - 2
                : 0
              : itemPos.y < 0.5
              ? absItemPos.y + 2
              : absItemPos.y - height - 2,
            width,
            height,
          };
        }
      }
    },
    [centerContainerVals, isMobile, selectedItemDescriptionPos, selectedItem]
  );

  useEffect(() => {
    (async () => {
      if (selectedItem) {
        const aspect = await getImageAspectRatio(
          selectedItem.image?.image?.data?.attributes?.url
        );
        if (aspect) {
          const pos: PosAndDim2D | undefined = await getImagePos(
            selectedItem.position,
            aspect
          );
          if (pos) {
            setSelectedItemImagePos(pos);
          }
        }
      }
    })();
  }, [selectedItem, getImagePos]);

  console.log(selectedItemImagePos);

  const handleIconClick = useCallback(
    (item: HomepageItem) => {
      if (selectedItemTitle) {
        setSelectedItemTitle(null);
      } else {
        setSelectedItemTitle(item.title);
      }
    },
    [selectedItemTitle]
  );

  const handleClickOffIcon = useCallback(() => {
    clearRectAnimation();
    setSelectedItemTitle(null);
  }, [clearRectAnimation]);

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
    }
  }, [dispatch]);

  // useEffect(() => {
  //   if (selectedYear && clearRectAnimation) {
  //     clearRectAnimation();
  //     setSelectedItemTitle(null);
  //   }
  // }, [selectedItem, selectedYear, clearRectAnimation]);

  const handleNavigate = useCallback(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
    }
    clearRectAnimation();
  }, [dispatch, clearRectAnimation]);

  useOnNavigate(handleNavigate);

  return (
    <>
      {centerContainerVals && shouldMount && (
        <MotionGridChild
          {...DEFAULT_ANIMATE_MODE}
          {...centerContainerVals}
          className=""
        >
          {parsedHomepageData?.data.attributes.items.map((item) => {
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
                    selectedItemTitle
                      ? item.title === selectedItemTitle
                        ? ""
                        : "pointer-events-none"
                      : ""
                  }`}
                >
                  <Icon
                    strokeWidth={4}
                    selected={selectedItemTitle === item.title}
                    width={
                      cellSize
                        ? cellSize?.x * (item.iconScale ? item.iconScale : 1)
                        : 1
                    }
                    height={
                      cellSize
                        ? cellSize?.y * (item.iconScale ? item.iconScale : 1)
                        : 1
                    }
                    className={`absolute icon-hover-glow hover:scale-125 transition-all duration-500
                      ${
                        selectedItemTitle
                          ? item.title === selectedItemTitle
                            ? "stroke-highlight"
                            : "stroke-landingIconInactive blur-[2px] "
                          : "stroke-white"
                      } ${
                      (selectedFilterType === null ||
                        selectedFilterType === item.type) &&
                      (selectedYear === null ||
                        String(selectedYear) === String(item.year))
                        ? selectedYear !== null
                          ? "transition-all scale-110"
                          : ""
                        : "!stroke-landingIconInactive blur-[1px]"
                    }
                    `}
                    style={{
                      animation: `flicker ${
                        Math.random() * 20 + 5
                      }s linear infinite`,
                      animationDelay: `${Math.random() * 2}s`,
                    }}
                  />
                </button>
              </GridChild>
            );
          })}
          {/* Selected Item Image & Click Mask*/}
          <AnimatePresence>
            {selectedItem && selectedItemImagePos && (
              <>
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
                      {selectedItemImagePos && (
                        <Image
                          src={selectedItem.image?.image?.data?.attributes?.url}
                          className="object-cover w-full h-full slide-in"
                          alt=""
                          width={selectedItemImagePos.width * TARGET_CELL_SIZE}
                          height={
                            selectedItemImagePos.height * TARGET_CELL_SIZE
                          }
                        />
                      )}
                    </Link>
                  </button>
                </MotionGridChild>
              </>
            )}
          </AnimatePresence>
          {selectedItem &&
            selectedItemDescriptionPos &&
            selectedItemImagePos && (
              <>
                <GridChild
                  {...selectedItemDescriptionPos}
                  isGrid={false}
                  className="relative text-offWhite"
                >
                  <div
                    className={`absolute top-0 left-0 w-full text-left bg-black transition-all duration-500 hover:scale-[102%]`}
                  >
                    <Link
                      href={`/project/${selectedItem.slug}`}
                      className="flex h-full w-full p-2"
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
                        <div className="flex text-offWhite items-center justify-center w-full h-full animate-arrowGesture animation-delay-500">
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
      {gridDim && selectedItemTitle && (
        <GridChild
          width={gridDim.x}
          height={gridDim.y}
          x={0}
          y={0}
          className="testinggg"
          onClick={() => {
            handleClickOffIcon();
          }}
        />
      )}
      <FooterRight
        footerRightHeight={4}
        footerRightWidth={6}
        isMounted={shouldMount}
        mobileTitleComponent={
          <div className="flex items-center">
            {selectedFilterType ? (
              (() => {
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
                  </>
                );
              })()
            ) : (
              <TfiLayoutMenuV color="white" className="mr-1" />
            )}

            <p>legend</p>
          </div>
        }
      >
        <GridChild
          x={0}
          y={0}
          width={6}
          height={4}
          className="w-full h-full text-[12px] text-offWhite"
        >
          <GridChild
            x={0}
            y={0}
            width={6}
            height={1}
            isGrid={false}
            className="border-b-[1px] border-offWhite flex justify-between items-center"
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
                      className={`w-4 mr-2 translate-y-[2px] ${
                        type === selectedFilterType ? "glow" : ""
                      }`}
                      stroke="white"
                      selected={false}
                    />
                    <p
                      className={`transition-all duration-500 ${
                        type === selectedFilterType ? "translate-x-2 glow" : ""
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
