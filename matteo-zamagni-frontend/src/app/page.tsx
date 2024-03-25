"use client";
import { Layout } from "@/components/Layout";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import { useEffect } from "react";

export default function Home() {
  const dispatch = useGlobalContextDispatch();

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
      dispatch({ type: "SET_IS_DARK", val: true });
    }
  }, [dispatch]);

  return <Layout footerRightHeight={5} footerRightWidth={7}></Layout>;
}
