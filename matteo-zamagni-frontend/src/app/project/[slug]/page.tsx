"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild } from "@/components/GridChild";
import { ImageGallery } from "@/components/ImageGallery";
import { MotionGridChild } from "@/components/MotionGridChild";
import { VideoPlayer } from "@/components/VideoPlayer";
import {
  clearGrid,
  drawVerticalLine,
  getCirclePoints,
  lightPixels,
} from "@/helpers/gridHelpers";
import { useGridLineAnimation } from "@/hooks/useGridAnimation";
import { useLEDScrollbar } from "@/hooks/useLEDScrollbar";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Grid, PosAndDim2D } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MdClose } from "react-icons/md";

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
  TEXT = "text",
  IMAGES = "images",
  VIDEO = "video",
}

const DUMMY_DATA: ProjectData = {
  text: {
    text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin hendrerit lorem et felis condimentum elementum non a nunc. Duis maximus nunc sit amet nisl cursus, sit amet efficitur purus rutrum. Pellentesque egestas egestas velit nec posuere. In faucibus dui ut placerat viverra. Nam sollicitudin aliquam orci id egestas. Morbi tempus euismod porttitor. Donec fringilla euismod lectus, condimentum laoreet mauris fermentum vel. Donec molestie gravida scelerisque. Aenean scelerisque egestas mauris sed sagittis. Praesent metus eros, cursus ac eleifend eu, lobortis at dolor. Mauris magna lacus, egestas eget est vitae, blandit laoreet quam. Nam quis faucibus eros. Morbi consequat est in libero congue consequat. Cras placerat nibh eget ligula luctus dignissim. Ut pretium nisi nunc, eget condimentum libero cursus ac. Morbi sed purus imperdiet, iaculis lacus sit amet, accumsan quam. In lorem metus, finibus in sodales eu, aliquam ut nibh. Vivamus sagittis, mi sed tristique vehicula, metus tellus viverra arcu, ut ultrices orci ante non elit. Maecenas interdum, ante non posuere consequat, metus nisl varius urna, id mattis dolor tortor eu augue. Cras nec efficitur dui. Quisque eu ex odio. Donec pretium bibendum mi porttitor ultricies. Pellentesque vehicula sapien in ex scelerisque, at vehicula libero venenatis. Proin ullamcorper ullamcorper ligula, nec vestibulum purus blandit facilisis. In mattis rutrum justo et posuere. Nulla elementum imperdiet mi, et condimentum libero cursus eget. In ac justo ac metus consequat viverra. Cras rutrum leo at venenatis scelerisque.",
  },
  images: [
    {
      thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
      imageURL: "https://placehold.co/640x480/EEE/ff0000/webp",
      alt: "image description",
    },
    {
      thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
      imageURL: "https://placehold.co/640x480/EEE/ff0000/webp",
      alt: "image description",
    },
    {
      thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
      imageURL: "https://placehold.co/640x480/EEE/ff0000/webp",
      alt: "image description",
    },
    // {
    //   thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
    //   imageURL: "https://placehold.co/640x480/EEE/ff0000",
    //   alt: "image description",
    // },
    // {
    //   thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
    //   imageURL: "https://placehold.co/640x480/EEE/ff0000",
    //   alt: "image description",
    // },
    // {
    //   thumbnailURL: "https://placehold.co/100x100/EEE/31343C",
    //   imageURL: "https://placehold.co/640x480/EEE/ff0000",
    //   alt: "image description",
    // },
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
  const [activeImageIndex, setActiveImageIndex] = useState<number | null>(null);

  const {
    startAnimation: startCircleAnimation,
    cancelAnimation: cancelCircleAnimation,
  } = useGridLineAnimation();

  const textCenterCellPos = useMemo(() => {
    const width =
      Math.floor(gridDim.x * 0.5 * CENTER_CELL_WIDTH_PROPOPRTION) * 2;
    const height =
      Math.floor(gridDim.y * 0.5 * CENTER_CELL_HEIGHT_PROPORTION) * 2;

    const yCenterOffest = Math.floor(gridDim.y * CENTER_CELL_OFFSET_PROPORTION);

    return {
      x: 1 + gridDim.x / 2 - width / 2,
      y: 1 + gridDim.y / 2 + yCenterOffest - height / 2,
      width,
      height,
    };
  }, [gridDim]);

  const textElementRef = useRef<HTMLDivElement | null>(null);
  useLEDScrollbar(
    textCenterCellPos.y,
    textCenterCellPos.y + textCenterCellPos.height - 1,
    textCenterCellPos.x + textCenterCellPos.width + 1,
    textElementRef
  );

  const updateLEDs = useCallback(
    (mode: ProjectMode) => {
      cancelCircleAnimation();
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
    [dispatch, grid, gridDim, cancelCircleAnimation, textCenterCellPos]
  );

  const handleChangeProjectMode = useCallback(
    (mode: ProjectMode) => {
      setProjectMode(mode);
      updateLEDs(mode);
      if (!ledIsSet) {
        setLedIsSet(true);
      }
    },
    [updateLEDs, ledIsSet]
  );

  const imageGridPositions: PosAndDim2D[] | null = useMemo(() => {
    if (projectMode !== ProjectMode.IMAGES) {
      return null;
    }

    const WIDTH = 4;
    const HEIGHT = 4;

    const imageCoords = getCirclePoints(
      { x: 0.5, y: 0.5 },
      0.3,
      DUMMY_DATA.images.length,
      gridDim
    ).map((coord) => ({
      x: Math.round(coord.x - WIDTH / 2),
      y: Math.round(coord.y - HEIGHT / 2),
      width: WIDTH,
      height: HEIGHT,
    }));

    return imageCoords;
  }, [projectMode, gridDim]);

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
    if (projectMode !== ProjectMode.IMAGES) {
      setActiveImageIndex(null);
    }
  }, [projectMode]);

  // Retrigger LEDs setting function when
  useEffect(() => {
    if (!ledIsSet) {
      handleChangeProjectMode(projectMode);
    }
  }, [ledIsSet, projectMode, handleChangeProjectMode]);

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "SET_IS_DARK", val: true });
    }
  }, [dispatch]);

  return (
    <>
      {/* Text View */}
      <AnimatePresence>
        {projectMode === ProjectMode.TEXT && (
          <GridChild className="" {...textCenterCellPos} isGrid={false}>
            <motion.div
              ref={textElementRef}
              initial={{ opacity: 0 }}
              exit={{ opacity: 0, transition: { delay: 0 } }}
              animate={{ opacity: 1 }}
              transition={{ type: "ease-in-out", duration: 0.5, delay: 0.5 }}
              key={projectMode}
              className="w-full h-full overflow-auto bg-black no-scrollbar"
            >
              {DUMMY_DATA.text.text}
            </motion.div>
          </GridChild>
        )}
      </AnimatePresence>
      {/* Image View */}
      <AnimatePresence>
        {projectMode === ProjectMode.IMAGES && imageGridPositions && (
          <>
            {imageGridPositions.map((imagePos, index) => (
              <MotionGridChild
                key={`${DUMMY_DATA.images[index].imageURL}-${index}`}
                initial={{ opacity: 0 }}
                exit={{ opacity: 0, transition: { delay: 0 } }}
                animate={{ opacity: 1 }}
                transition={{ type: "ease-in-out", duration: 0.5, delay: 0.5 }}
                className=""
                {...imagePos}
                isGrid={false}
              >
                <button
                  className="w-full h-full bg-red-500"
                  onClick={() => {
                    setActiveImageIndex(index);
                  }}
                ></button>
              </MotionGridChild>
            ))}
          </>
        )}
      </AnimatePresence>
      {/* Image Gallery View */}
      <AnimatePresence>
        {activeImageIndex !== null ? (
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
              images={DUMMY_DATA.images}
              initialSlide={activeImageIndex}
              handleClose={() => setActiveImageIndex(null)}
            />
          </MotionGridChild>
        ) : null}
      </AnimatePresence>
      {/* Video Player View */}
      <AnimatePresence>
        {projectMode === ProjectMode.VIDEO ? (
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
              url={"https://vimeo.com/203563626"}
              handleClose={() => handleChangeProjectMode(ProjectMode.TEXT)}
            />
          </MotionGridChild>
        ) : null}
      </AnimatePresence>

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
                  className={`icon-hover-glow duration-500 transition-all text-[12px] block ${
                    projectMode === ProjectMode.TEXT
                      ? "text-white"
                      : "text-textInactive"
                  }`}
                  onClick={() => {
                    handleChangeProjectMode(ProjectMode.TEXT);
                  }}
                >
                  text
                </button>
                <button
                  className={`icon-hover-glow duration-500 transition-all text-[12px] block ${
                    projectMode === ProjectMode.IMAGES
                      ? "text-white"
                      : "text-textInactive"
                  }`}
                  onClick={() => {
                    handleChangeProjectMode(ProjectMode.IMAGES);
                  }}
                >
                  images
                </button>
                <button
                  className={`icon-hover-glow duration-500 transition-all text-[12px] block ${
                    projectMode === ProjectMode.VIDEO
                      ? "text-white"
                      : "text-textInactive"
                  }`}
                  onClick={() => {
                    handleChangeProjectMode(ProjectMode.VIDEO);
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