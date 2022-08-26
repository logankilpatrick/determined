import yaml from 'js-yaml';
import React, { useEffect, useRef } from 'react';

import { useSetDynamicTabBar } from 'components/DynamicTabs';
import Grid, { GridMode } from 'components/Grid';
import LearningCurveChart from 'components/LearningCurveChart';
import Page from 'components/Page';
import Section from 'components/Section';
import { InteractiveTableSettings } from 'components/Table/InteractiveTable';
import { SyncProvider } from 'components/UPlot/SyncableBounds';
import useSettings from 'hooks/useSettings';
import TrialTable from 'pages/TrialsComparison/Table/TrialTable';
import { V1AugmentedTrial } from 'services/api-ts-sdk';
import Message, { MessageType } from 'shared/components/Message';
import { Scale } from 'types';
import { metricToKey } from 'utils/metric';

import useHighlight from '../../hooks/useHighlight';

import useTrialActions from './Actions/useTrialActions';
import {
  useTrialCollections,
} from './Collections/useTrialCollections';
import useLearningCurveData from './Metrics/useLearningCurveData';
import { trialsTableSettingsConfig } from './Table/settings';
import { useFetchTrials } from './Trials/useFetchTrials';
import css from './TrialsComparison.module.scss';
const initData = [ [] ];
interface Props {
  projectId: string;
}

const TrialsComparison: React.FC<Props> = ({ projectId }) => {

  const tableSettingsHook = useSettings<InteractiveTableSettings>(trialsTableSettingsConfig);

  const refetcher = useRef<() => void>();

  const C = useTrialCollections(projectId, tableSettingsHook, refetcher);

  const { settings: tableSettings } = tableSettingsHook;

  const { trials, refetch } = useFetchTrials({
    filters: C.filters,
    limit: tableSettings.tableLimit,
    offset: tableSettings.tableOffset,
    sorter: C.sorter,
  });

  useEffect(() => refetcher.current = refetch, [ refetch ]);

  const A = useTrialActions({
    filters: C.filters,
    openCreateModal: C.openCreateModal,
    refetch,
    sorter: C.sorter,
  });

  // console.log(yaml.dump(C.filters));

  const highlights = useHighlight((trial: V1AugmentedTrial): number => trial.trialId);

  const containerRef = useRef<HTMLElement>(null);

  const chartSeries = useLearningCurveData(trials.ids, trials.metrics, trials.maxBatch);

  useSetDynamicTabBar(C.controls);

  return (
    <Page className={css.base} containerRef={containerRef}>
      <Section bodyBorder bodyScroll>
        <div className={css.container}>
          <div className={css.chart}>
            {trials.metrics.length === 0 ? (
              <Message title="No Metrics for Selected Trials" type={MessageType.Empty} />
            ) : (
              <Grid
                border={true}
                //  TODO: use screen size
                minItemWidth={600}
                mode={GridMode.AutoFill}>
                <SyncProvider>
                  {trials.metrics.map((metric) => (
                    <LearningCurveChart
                      data={chartSeries?.metrics?.[metricToKey(metric)] ?? initData}
                      focusedTrialId={highlights.id}
                      key={metricToKey(metric)}
                      selectedMetric={metric}
                      selectedScale={Scale.Linear}
                      selectedTrialIds={A.selectedTrials}
                      trialIds={trials.ids}
                      xValues={chartSeries?.batches ?? []}
                      onTrialFocus={highlights.focus}
                    />
                  ))}
                </SyncProvider>
              </Grid>
            )}
          </div>
          {A.dispatcher}
          <TrialTable
            actionsInterface={A}
            collectionsInterface={C}
            containerRef={containerRef}
            highlights={highlights}
            tableSettingsHook={tableSettingsHook}
            trialsWithMetadata={trials}
          />
        </div>
      </Section>
      {A.modalContextHolder}
      {C.modalContextHolder}
    </Page>
  );
};

export default TrialsComparison;
