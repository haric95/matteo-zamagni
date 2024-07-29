"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild } from "@/components/GridChild";
import { ImageGallery } from "@/components/ImageGallery";
import { MotionGridChild } from "@/components/MotionGridChild";
import { VideoPlayer } from "@/components/VideoPlayer";
import { DEFAULT_ANIMATE_MODE } from "@/const";
import {
  clearGrid,
  drawVerticalLine,
  getCirclePoints,
  lightPixels,
} from "@/helpers/gridHelpers";
import { parseTagsString } from "@/helpers/parseTagsString";
import { useIsMobile } from "@/hooks/useIsMobile";
import { useLEDScrollbar } from "@/hooks/useLEDScrollbar";
import { usePrefetchImages } from "@/hooks/usePrefetchImages";
import { StrapiImageResponse, useStrapi } from "@/hooks/useStrapi";
import { useTheme } from "@/hooks/useTheme";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Grid, HomepageData, PosAndDim2D } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import { useCallback, useEffect, useMemo, useState } from "react";
import { TfiLayoutMenuV } from "react-icons/tfi";
import Markdown from "react-markdown";
import { TypeAnimation } from "react-type-animation";

const CENTER_CELL_PADDING_X = 16;
const CENTER_CELL_PADDING_Y = 6;
const CENTER_CELL_Y_OFFSET = 2;
const CENTER_CELL_PADDING_X_MOBILE = 2;
const CENTER_CELL_PADDING_Y_MOBILE = 8;
const CENTER_CELL_Y_OFFSET_MOBILE = 2;

enum ProjectMode {
  TEXT = "text",
  IMAGES = "images",
  VIDEO = "video",
}

type ProjectPageData = {
  title: string;
  slug: string;
  text?: string;
  images: {
    image: StrapiImageResponse;
    thumbnail?: StrapiImageResponse;
    alt?: string;
  }[];
  videoURL?: string;
};

// TODO: Add on mount delay to wait until bg color change has happened
// TODO: Add About Modes
export default function Project({ params }: { params: { slug: string } }) {
  const projectData = useStrapi<ProjectPageData, true>("/projects", {
    "filters[slug][$eq]": params.slug,
    populate: "deep",
  });
  const projectItem = projectData?.data[0];
  const homepageData = useStrapi<HomepageData, false>("/homepage", {
    populate: "deep",
  });
  const homepageItem = homepageData?.data.attributes.items.find(
    (homepageItem) => homepageItem.slug === params.slug
  );
  const isMobile = useIsMobile();
  const { shouldMount } = useTheme({ isDark: true });

  usePrefetchImages(
    projectItem?.attributes.images.map(
      (image) => image.image.data.attributes.url
    ) || null
  );
  usePrefetchImages(
    projectItem?.attributes.images.map(
      (image) =>
        image.thumbnail?.data.attributes.url || image.image.data.attributes.url
    ) || null
  );

  const { gridDim, grid } = useGlobalContext() as {
    gridDim: Dim2D;
    grid: Grid;
  };
  const dispatch = useGlobalContextDispatch();

  const [projectMode, setProjectMode] = useState<ProjectMode | null>(null);
  const [ledIsSet, setLedIsSet] = useState(false);
  const [activeImageIndex, setActiveImageIndex] = useState<number | null>(null);
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(
    null
  );

  useEffect(() => {
    if (projectItem) {
      if (projectItem.attributes.videoURL) {
        setProjectMode(ProjectMode.VIDEO);
      } else {
        setProjectMode(ProjectMode.TEXT);
      }
    }
  }, [projectItem]);

  const textCenterCellPos = useMemo<PosAndDim2D>(() => {
    const width = isMobile
      ? gridDim.x - CENTER_CELL_PADDING_X_MOBILE * 2
      : gridDim.x - CENTER_CELL_PADDING_X * 2;
    const height = isMobile
      ? gridDim.y - CENTER_CELL_PADDING_Y_MOBILE * 2
      : gridDim.y - CENTER_CELL_PADDING_Y * 2;

    const yOffset = isMobile
      ? CENTER_CELL_Y_OFFSET_MOBILE
      : CENTER_CELL_Y_OFFSET;

    return {
      x: isMobile ? CENTER_CELL_PADDING_X_MOBILE : CENTER_CELL_PADDING_X,
      y:
        (isMobile ? CENTER_CELL_PADDING_Y_MOBILE : CENTER_CELL_PADDING_Y) +
        yOffset,
      width,
      height,
    };
  }, [gridDim, isMobile]);

  const titleCellPos = useMemo(() => {
    const width = isMobile
      ? textCenterCellPos.width
      : Math.floor((gridDim.x - textCenterCellPos.width) / 2) - 4;
    const height = isMobile ? 2 : 4;
    return {
      x: isMobile ? textCenterCellPos.x - 1 : 2,
      y: isMobile ? 5 : gridDim.y / 2 - height / 2,
      width,
      height,
    };
  }, [gridDim, textCenterCellPos, isMobile]);

  const tagsCellPos = useMemo(() => {
    const width = isMobile
      ? textCenterCellPos.width
      : Math.floor((gridDim.x - textCenterCellPos.width) / 2) - 6;
    const height = isMobile ? 2 : 4;
    return {
      x: isMobile
        ? textCenterCellPos.x - 1
        : textCenterCellPos.x + textCenterCellPos.width + 3,
      y: isMobile
        ? titleCellPos.y + titleCellPos.height
        : gridDim.y / 2 - height / 2,
      width,
      height,
    };
  }, [gridDim, textCenterCellPos, isMobile, titleCellPos]);

  const handleChangeTextElement = useCallback((div: HTMLDivElement | null) => {
    if (HTMLDivElement) {
      setScrollElement(div);
    }
  }, []);

  useLEDScrollbar(
    textCenterCellPos.y,
    textCenterCellPos.y + textCenterCellPos.height - 1,
    textCenterCellPos.x + textCenterCellPos.width + 1,
    scrollElement
  );

  const updateLEDs = useCallback(
    (mode: ProjectMode) => {
      if (dispatch) {
        const clearedGrid = clearGrid(grid);
        if (mode === ProjectMode.TEXT) {
          const updatedGrid = drawVerticalLine(
            drawVerticalLine(
              clearedGrid,
              // grid is 0 indexed and we want to highlight the column on the outside of the box
              textCenterCellPos.x - 1,
              textCenterCellPos.y,
              textCenterCellPos.y + textCenterCellPos.height
            ),
            textCenterCellPos.x + textCenterCellPos.width,
            textCenterCellPos.y,
            textCenterCellPos.y + textCenterCellPos.height
          );
          dispatch({ type: "UPDATE_GRID", grid: updatedGrid });
        } else if (mode === ProjectMode.IMAGES) {
          const circlePoints = getCirclePoints({ x: 0.5, y: 0.5 }, 0.1, 8, {
            x: gridDim.x - 1,
            y: gridDim.y - 1,
          });
          const updatedGrid = lightPixels(clearedGrid, circlePoints);
          dispatch({ type: "UPDATE_GRID", grid: updatedGrid });
        } else if (mode === ProjectMode.VIDEO) {
          dispatch({ type: "CLEAR_GRID" });
        }
      }
    },
    [dispatch, grid, gridDim, textCenterCellPos]
  );

  const handleChangeProjectMode = useCallback(
    (mode: ProjectMode | null) => {
      if (mode) {
        setProjectMode(mode);
        updateLEDs(mode);
        if (!ledIsSet) {
          setLedIsSet(true);
        }
      }
    },
    [updateLEDs, ledIsSet]
  );

  const imageGridPositions: PosAndDim2D[] | null = useMemo(() => {
    if (!projectItem || !projectItem.attributes.images.length) {
      return null;
    }

    const WIDTH = 4;
    const HEIGHT = 4;

    const imageCoords = getCirclePoints(
      { x: 0.5, y: 0.5 },
      0.3,
      projectItem.attributes.images.length,
      gridDim
    ).map((coord) => ({
      x: Math.round(coord.x - WIDTH / 2),
      y: Math.round(coord.y - HEIGHT / 2),
      width: WIDTH,
      height: HEIGHT,
    }));

    return imageCoords;
  }, [gridDim, projectItem]);

  const galleryGridPosition = useMemo<PosAndDim2D>(() => {
    const GALLERY_PADDING_X = 12;
    const GALLERY_PADDING_Y = 4;
    return {
      x: GALLERY_PADDING_X,
      y: GALLERY_PADDING_Y + 1,
      width: gridDim.x - GALLERY_PADDING_X * 2,
      height: gridDim.y - GALLERY_PADDING_Y * 2,
    };
  }, [gridDim]);

  // Reset LEDs on gridDim change
  useEffect(() => {
    if (gridDim) {
      setLedIsSet(false);
    }
  }, [gridDim]);

  useEffect(() => {
    if (!ledIsSet) {
      handleChangeProjectMode(projectMode);
    }
  }, [ledIsSet, handleChangeProjectMode, projectMode]);

  useEffect(() => {
    if (projectMode !== ProjectMode.IMAGES) {
      setActiveImageIndex(null);
    }
  }, [projectMode]);

  useTheme({ isDark: true });

  return (
    <>
      <MotionGridChild
        {...DEFAULT_ANIMATE_MODE}
        transition={{ type: "ease-in-out", duration: 0.5, delay: 0.5 }}
        exit={{ opacity: 0, transition: { duration: 0.5, delay: 0 } }}
        className="flex justify-start md:justify-center items-center bg-black text-white"
        {...titleCellPos}
        isGrid={false}
      >
        <div className="w-full text-left md:text-center w-full flex md:justify-center">
          {projectItem && (
            <TypeAnimation
              sequence={[projectItem?.attributes.title]}
              wrapper="span"
              speed={50}
              style={{ display: "inline-block" }}
              className="text-lg"
              cursor={false}
            />
          )}
          {/* <h2 className="w-fit text-lg p-4">
                  {projectItem?.attributes.title}
                </h2> */}
        </div>
      </MotionGridChild>
      <MotionGridChild
        {...DEFAULT_ANIMATE_MODE}
        transition={{ type: "ease-in-out", duration: 0.5, delay: 0.5 }}
        exit={{ opacity: 0, transition: { duration: 0.5, delay: 0 } }}
        isGrid={false}
        className="flex justify-center items-center bg-black"
        {...tagsCellPos}
      >
        <div
          className={`w-full md:text-center md:p-8 w-full flex justify-start md:justify-center`}
        >
          <div className="w-fit text-s md:text-lg md:p-4 flex md:block">
            {homepageItem &&
              (isMobile ? (
                <div>
                  <TypeAnimation
                    sequence={[parseTagsString(homepageItem.tags).join(", ")]}
                    wrapper="span"
                    speed={1}
                    style={{ display: "inline-block" }}
                    className="text-s"
                    cursor={false}
                  />
                </div>
              ) : (
                parseTagsString(homepageItem?.tags).map((tag) => (
                  <div key={tag}>
                    <TypeAnimation
                      sequence={[500, tag]}
                      wrapper="span"
                      speed={1}
                      style={{ display: "inline-block" }}
                      className="text-lg"
                      cursor={false}
                    />
                  </div>
                ))
              ))}
          </div>
        </div>
      </MotionGridChild>
      {/* Text View */}
      <AnimatePresence>
        {projectMode === ProjectMode.TEXT && (
          <>
            <GridChild className="" {...textCenterCellPos} isGrid={false}>
              <motion.div
                ref={handleChangeTextElement}
                initial={{ opacity: 0 }}
                exit={{ opacity: 0, transition: { delay: 0 } }}
                animate={{ opacity: 1 }}
                transition={{ type: "ease-in-out", duration: 0.5, delay: 0.5 }}
                key={projectMode}
                style={{ whiteSpace: "break-spaces" }}
                className="w-full h-full overflow-auto bg-black no-scrollbar"
              >
                <Markdown className={"markdown"}>
                  {projectItem?.attributes.text}
                </Markdown>
              </motion.div>
            </GridChild>
          </>
        )}
      </AnimatePresence>
      {/* Image View */}
      <AnimatePresence>
        {projectMode === ProjectMode.IMAGES &&
          imageGridPositions &&
          projectItem && (
            <>
              {imageGridPositions.map((imagePos, index) => (
                <MotionGridChild
                  key={`${projectItem.attributes.images[index].image.data.attributes.url}-${index}`}
                  initial={{ opacity: 0 }}
                  exit={{ opacity: 0, transition: { delay: 0 } }}
                  animate={{ opacity: 1 }}
                  transition={{
                    type: "ease-in-out",
                    duration: 0.5,
                    delay: 0.5,
                  }}
                  className="image-hover-glow hover:scale-105 transition-all duration-500"
                  {...imagePos}
                  isGrid={false}
                >
                  <button
                    className="w-full h-full relative"
                    onClick={() => {
                      setActiveImageIndex(index);
                    }}
                  >
                    <Image
                      src={
                        projectItem.attributes.images[index].image.data
                          .attributes.url
                      }
                      alt={""}
                      layout="fill"
                      objectFit="cover"
                    />
                  </button>
                </MotionGridChild>
              ))}
            </>
          )}
      </AnimatePresence>
      {/* Image Gallery View */}
      <AnimatePresence>
        {activeImageIndex !== null && projectItem ? (
          <MotionGridChild
            initial={{ opacity: 0 }}
            exit={{ opacity: 0, transition: { delay: 0 } }}
            animate={{ opacity: 1 }}
            transition={{ type: "ease-in-out", duration: 0.5 }}
            {...galleryGridPosition}
            className="z-10 relative"
            isGrid={false}
          >
            <ImageGallery
              images={projectItem.attributes.images.map((image) => ({
                imageURL: image.image.data.attributes.url,
                alt: image.alt ?? "",
              }))}
              initialSlide={activeImageIndex}
              handleClose={() => setActiveImageIndex(null)}
            />
          </MotionGridChild>
        ) : null}
      </AnimatePresence>
      {/* Video Player View */}
      <AnimatePresence>
        {projectMode === ProjectMode.VIDEO &&
        projectItem?.attributes.videoURL ? (
          <MotionGridChild
            initial={{ opacity: 0 }}
            exit={{ opacity: 0, transition: { delay: 0 } }}
            animate={{ opacity: 1 }}
            transition={{ type: "ease-in-out", duration: 0.5 }}
            {...galleryGridPosition}
            className="z-10 relative"
            isGrid={false}
          >
            <VideoPlayer
              url={projectItem.attributes.videoURL}
              handleClose={() => handleChangeProjectMode(ProjectMode.TEXT)}
            />
          </MotionGridChild>
        ) : null}
      </AnimatePresence>

      {shouldMount && (
        <FooterRight
          footerRightHeight={5}
          footerRightWidth={6}
          mobileTitleComponent={
            <div className="flex items-center">
              <TfiLayoutMenuV color="white" className="mr-1" />
              <p>navigation - {projectMode}</p>
            </div>
          }
        >
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
                    className={`icon-hover-glow duration-500 transition-all text-[12px] block ${
                      projectMode === ProjectMode.TEXT
                        ? "text-white translate-x-2"
                        : "text-textInactive"
                    }`}
                    onClick={() => {
                      handleChangeProjectMode(ProjectMode.TEXT);
                      if (dispatch) {
                        dispatch({
                          type: "SET_MOBILE_FOOTER_MENU",
                          isOpen: false,
                        });
                      }
                    }}
                  >
                    text
                  </button>
                  <button
                    className={`icon-hover-glow duration-500 transition-all text-[12px] block ${
                      projectMode === ProjectMode.IMAGES
                        ? "text-white translate-x-2"
                        : "text-textInactive"
                    }`}
                    onClick={() => {
                      handleChangeProjectMode(ProjectMode.IMAGES);
                      if (dispatch) {
                        dispatch({
                          type: "SET_MOBILE_FOOTER_MENU",
                          isOpen: false,
                        });
                      }
                    }}
                  >
                    images
                  </button>
                  {projectItem?.attributes.videoURL ? (
                    <button
                      className={`icon-hover-glow duration-500 transition-all text-[12px] block ${
                        projectMode === ProjectMode.VIDEO
                          ? "text-white"
                          : "text-textInactive"
                      }`}
                      onClick={() => {
                        handleChangeProjectMode(ProjectMode.VIDEO);
                        if (dispatch) {
                          dispatch({
                            type: "SET_MOBILE_FOOTER_MENU",
                            isOpen: false,
                          });
                        }
                      }}
                    >
                      video
                    </button>
                  ) : (
                    <div className="h-[18px]" />
                  )}
                </div>
              </div>
            </div>
          </div>
        </FooterRight>
      )}
    </>
  );
}
