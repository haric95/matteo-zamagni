"use client";
import { FooterRight } from "@/components/FooterRight";
import { GridChild } from "@/components/GridChild";
import { Plus, TriangleDown } from "@/components/Icons";
import { useGlobalContextDispatch } from "@/state/GlobalStore";
import { useEffect } from "react";

export default function Home() {
  const dispatch = useGlobalContextDispatch();

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
      dispatch({ type: "SET_IS_DARK", val: true });
    }
  }, [dispatch]);

  return (
    <>
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
              <div className="w-full h-4 flex">
                <Plus className="w-4 mr-4 translate-y-[2px]" stroke="white" />
                <p>projects</p>
              </div>
              <div className="w-full h-4 flex">
                <TriangleDown
                  className="w-4 mr-4 translate-y-[2px]"
                  stroke="white"
                />
                <p>exhibitions</p>
              </div>
            </div>
          </GridChild>
        </GridChild>
      </FooterRight>
    </>
  );
}
