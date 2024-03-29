"use client";
import { FooterRight } from "@/components/FooterRight";
import {
  BackChevrons,
  Circle,
  HorizontalLines,
  Plus,
  Star,
  TriangleDown,
} from "@/components/Icons";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Grid } from "@/types/global";
import { useEffect, useState } from "react";

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

// TODO: Add on mount delay to wait until bg color change has happened
// TODO: Add About Modes
export default function Index() {
  const { gridDim, grid } = useGlobalContext() as {
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

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "SET_IS_DARK", val: false });
      dispatch({ type: "CLEAR_GRID" });
    }
  }, [dispatch]);

  return (
    <>
      <FooterRight footerRightHeight={8} footerRightWidth={6}>
        <div
          className="grid col-span-full row-span-full  "
          style={{
            gridTemplateColumns: `repeat(${6}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${8}, minmax(0, 1fr))`,
          }}
        >
          <div className="col-span-full row-span-1 flex items-start border-black border-b-[1px]">
            <p className="text-[12px]">legend</p>
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
                        strokeWidth={4}
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
                {/* <WorkIndexTypeIcon.project
                  className="mr-2 w-4 h-4"
                  stroke="white"
                  strokeWidth={4}
                />
                <p className="translate-y-[-1px]">project</p> */}
                {/* <div className="w-full h-fit flex items-center w-2 h-2">
                  <WorkIndexTypeIcon.exhibition className="mr-2 w-2 h-2" />
                  <p className="translate-y-[-1px]">exhibition</p>
                </div>
                <div className="w-full h-fit flex items-center w-2 h-2">
                  <WorkIndexTypeIcon.installation className="mr-2 w-2 h-2" />
                  <p className="translate-y-[-1px]">installation</p>
                </div>
                <div className="w-full h-fit flex items-center w-2 h-2">
                  <WorkIndexTypeIcon.performance className="mr-2 w-2 h-2" />
                  <p className="translate-y-[-1px]">performance</p>
                </div>
                <div className="w-full h-fit flex items-center w-2 h-2">
                  <WorkIndexTypeIcon.film className="mr-2 w-2 h-2" />
                  <p className="translate-y-[-1px]">film</p>
                </div>
                <div className="w-full h-fit flex items-center w-2 h-2">
                  <WorkIndexTypeIcon.print className="mr-2 w-2 h-2" />
                  <p className="translate-y-[-1px]">print</p>
                </div> */}
              </div>
            </div>
          </div>
        </div>
      </FooterRight>
    </>
  );
}
