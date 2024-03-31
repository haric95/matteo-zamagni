"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild } from "@/components/GridChild";
import { HEADER_OFFSET_Y, TOTAL_HEADER_HEIGHT } from "@/components/Header";
import {
  IconComponent,
  Plus,
  SelectableIconComponent,
  TriangleDown,
} from "@/components/Icons";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { useEffect, useMemo, useState } from "react";

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
};

const DUMMY_HOMEPAGE_ITEMS: HomepageItem[] = [
  {
    position: { x: 0.2, y: 0.2 },
    type: HomepageItemType.EXHIBITION,
    title: "HELLO TESTING",
    year: "2022",
  },
  {
    position: { x: 0.4, y: 0.1 },
    type: HomepageItemType.PROJECT,
    title: "HELLO TESTING 2",
    year: "2021",
  },
];

export default function Home() {
  const dispatch = useGlobalContextDispatch();
  const { gridDim } = useGlobalContext();

  const [selectedItem, setSelectedItem] = useState<string | null>(null);

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
                    if (selectedItem === item.title) {
                      setSelectedItem(null);
                    } else {
                      setSelectedItem(item.title);
                    }
                  }}
                  className="testtt w-full h-full flex items-center justify-center"
                >
                  <Icon
                    stroke="white"
                    strokeWidth={4}
                    selected={selectedItem === item.title}
                  />
                </button>
              </GridChild>
            );
          })}
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
