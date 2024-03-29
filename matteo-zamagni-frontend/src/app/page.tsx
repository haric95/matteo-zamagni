"use client";
import { FooterRight } from "@/components/FooterRight";
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
      <FooterRight footerRightHeight={5} footerRightWidth={7}></FooterRight>
    </>
  );
}
