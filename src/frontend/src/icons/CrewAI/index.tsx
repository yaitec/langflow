import React, { forwardRef } from "react";
import SvgCrewAI from "./SvgCrewAI";

export const CrewAI = forwardRef<SVGSVGElement, React.PropsWithChildren<{}>>(
  (props, ref) => {
    return <SvgCrewAI className="icon" ref={ref} {...props} />;
  },
);
